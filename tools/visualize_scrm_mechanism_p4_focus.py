import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.core import YAMLConfig


DEFAULT_CONFIG = "configs/rtdetr/rtdetr_r50vd_6x_hixray_sc_xlce_soft.yml"
DEFAULT_CKPT = "output/rtdetr_r50vd_6x_hixray_sc_xlce_soft/checkpoint0071.pth"
DEFAULT_IMAGE = "dataset/hixray/images/val/006459201008393.jpg"
DEFAULT_ANN = "dataset/hixray/annotations/instances_val.json"
DEFAULT_OUT = "paper_eaai_records_20260912/figures/final_main_figures/scrm_mechanism_p4_focus"


def tensor_to_map(tensor, mode="abs_mean"):
    t = tensor.detach().float()
    if mode == "abs_mean":
        m = t.abs().mean(dim=1)[0]
    elif mode == "mean":
        m = t.mean(dim=1)[0]
    else:
        raise ValueError(mode)
    return m.cpu().numpy()


def robust_norm(x, p_low=1.0, p_high=99.0):
    x = np.asarray(x, dtype=np.float32)
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    if hi <= lo:
        return np.zeros_like(x)
    x = (x - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0)


def resize_map(arr, size=(640, 640)):
    t = torch.from_numpy(arr).float()[None, None]
    t = F.interpolate(t, size=size, mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


def load_gt_boxes(ann_file, file_name):
    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_info = None
    for im in data["images"]:
        if im["file_name"] == file_name:
            image_info = im
            break
    if image_info is None:
        return []

    image_id = image_info["id"]
    boxes = []
    for ann in data["annotations"]:
        if ann["image_id"] == image_id:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
    return boxes


def scale_boxes(boxes, orig_size, new_size=(640, 640)):
    ow, oh = orig_size
    nw, nh = new_size
    sx = nw / ow
    sy = nh / oh
    scaled = []
    for x1, y1, x2, y2 in boxes:
        scaled.append([x1 * sx, y1 * sy, x2 * sx, y2 * sy])
    return scaled


def make_input_with_gt(image_pil, boxes_scaled):
    out = image_pil.copy()
    draw = ImageDraw.Draw(out)
    for box in boxes_scaled:
        draw.rectangle(box, outline=(230, 30, 30), width=4)
    return out


def overlay_heatmap(image_np, response, cmap="turbo", alpha=0.48):
    response = np.clip(response, 0.0, 1.0)
    cm = plt.get_cmap(cmap)
    heat = cm(response)[..., :3]
    img = image_np.astype(np.float32) / 255.0
    blended = (1.0 - alpha) * img + alpha * heat
    return np.clip(blended, 0.0, 1.0)


def to_uint8(arr):
    if arr.dtype == np.uint8:
        return arr
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def draw_roi_on_array(arr, roi, color=(255, 215, 0), width=4):
    img = Image.fromarray(to_uint8(arr))
    draw = ImageDraw.Draw(img)
    draw.rectangle(roi, outline=color, width=width)
    return np.asarray(img)


def compute_union_roi(boxes_scaled, canvas_size=(640, 640), margin=28):
    if not boxes_scaled:
        w, h = canvas_size
        return [int(w * 0.65), int(h * 0.35), int(w * 0.95), int(h * 0.75)]

    x1 = min(b[0] for b in boxes_scaled)
    y1 = min(b[1] for b in boxes_scaled)
    x2 = max(b[2] for b in boxes_scaled)
    y2 = max(b[3] for b in boxes_scaled)

    x1 -= margin
    y1 -= margin
    x2 += margin
    y2 += margin

    w, h = canvas_size
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w - 1, int(x2))
    y2 = min(h - 1, int(y2))

    # prevent too tiny crop
    if (x2 - x1) < 120:
        pad = (120 - (x2 - x1)) // 2 + 1
        x1 = max(0, x1 - pad)
        x2 = min(w - 1, x2 + pad)
    if (y2 - y1) < 120:
        pad = (120 - (y2 - y1)) // 2 + 1
        y1 = max(0, y1 - pad)
        y2 = min(h - 1, y2 + pad)

    return [x1, y1, x2, y2]


def crop_roi(arr, roi):
    x1, y1, x2, y2 = roi
    return arr[y1:y2, x1:x2]


def main(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("Loading configuration...")
    cfg = YAMLConfig(args.config, resume=args.checkpoint)

    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
        print("Using EMA weights")
    else:
        state = checkpoint["model"]
        print("Using model weights")

    cfg.model.load_state_dict(state)
    model = cfg.model.to(device)
    model.eval()

    # locate P4 block only
    p4_block = None
    p4_name = None
    for name, module in model.named_modules():
        if module.__class__.__name__ == "SCXLCESoftBlock" and getattr(module, "scale_name", None) == "P4":
            p4_block = module
            p4_name = name
            break

    if p4_block is None:
        raise RuntimeError("P4 SCXLCESoftBlock not found.")

    print("Found P4 block:", p4_name)

    cache = {}

    def pre_hook(module, inputs):
        cache["x"] = inputs[0].detach()

    def xlce_hook(module, inputs, output):
        cache["xlce_out"] = output.detach()

    def logit_hook(module, inputs, output):
        cache["gate_logit"] = output.detach()

    h1 = p4_block.register_forward_pre_hook(pre_hook)
    h2 = p4_block.xlce.register_forward_hook(xlce_hook)
    h3 = p4_block.consistency_gate[0].register_forward_hook(logit_hook)

    image_path = Path(args.image)
    im_orig = Image.open(image_path).convert("RGB")
    orig_size = im_orig.size

    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    im_data = transform(im_orig)[None].to(device)

    print("Running forward pass...")
    with torch.no_grad():
        _ = model(im_data)

    h1.remove()
    h2.remove()
    h3.remove()

    x = cache["x"]
    xlce_out = cache["xlce_out"]
    gate_logit = cache["gate_logit"]

    residual = xlce_out - x
    gate = torch.sigmoid(gate_logit / p4_block.sc_gate_temperature)
    gate_soft = p4_block.sc_soft_base + (1.0 - p4_block.sc_soft_base) * gate
    gate_soft = p4_block.sc_gate_floor + (1.0 - p4_block.sc_gate_floor) * gate_soft
    modulated = p4_block.enhance_alpha * gate_soft * residual
    suppressed = p4_block.enhance_alpha * (1.0 - gate_soft) * residual

    print("P4 x shape         :", tuple(x.shape))
    print("P4 residual shape  :", tuple(residual.shape))
    print("P4 gate_soft shape :", tuple(gate_soft.shape))
    print("P4 modulated shape :", tuple(modulated.shape))
    print("P4 suppressed shape:", tuple(suppressed.shape))
    print("P4 gate range      :", float(gate_soft.min()), float(gate_soft.max()), float(gate_soft.mean()))

    input_640 = im_orig.resize((640, 640), Image.Resampling.BILINEAR)
    input_np = np.asarray(input_640)

    gt_boxes = load_gt_boxes(args.annotation, image_path.name)
    gt_boxes_scaled = scale_boxes(gt_boxes, orig_size, (640, 640))
    input_gt = make_input_with_gt(input_640, gt_boxes_scaled)

    roi = compute_union_roi(gt_boxes_scaled, canvas_size=(640, 640), margin=28)
    print("ROI:", roi)

    # response maps
    # --------------------------------------------------
    # Shared residual scale:
    # LEM / retained / attenuated residuals must be directly comparable.
    # Use the LEM residual as the common reference scale.
    # --------------------------------------------------
    lem_raw = tensor_to_map(residual, mode="abs_mean")
    mod_raw = tensor_to_map(modulated, mode="abs_mean")
    sup_raw = tensor_to_map(suppressed, mode="abs_mean")

    common_hi = np.percentile(lem_raw, 99.0)
    common_hi = max(float(common_hi), 1e-8)

    lem_map = np.clip(lem_raw / common_hi, 0.0, 1.0)
    mod_map = np.clip(mod_raw / common_hi, 0.0, 1.0)
    sup_map = np.clip(sup_raw / common_hi, 0.0, 1.0)

    lem_map = resize_map(lem_map, (640, 640))
    mod_map = resize_map(mod_map, (640, 640))
    sup_map = resize_map(sup_map, (640, 640))

    # SCM gate uses its true physical range, 0.5--1.0.
    gate_map_raw = tensor_to_map(gate_soft, mode="mean")
    gate_min = p4_block.sc_gate_floor + (1.0 - p4_block.sc_gate_floor) * p4_block.sc_soft_base
    gate_max = 1.0
    gate_map = np.clip(
        (gate_map_raw - gate_min) / max(gate_max - gate_min, 1e-8),
        0.0,
        1.0
    )
    gate_map = resize_map(gate_map, (640, 640))

    # overlays
    input_gt_np = np.asarray(input_gt)
    lem_overlay = to_uint8(overlay_heatmap(input_np, lem_map))
    # Pure SCM gate heatmap: do not blend with the X-ray image.
    gate_rgb = plt.get_cmap("turbo")(gate_map)[..., :3]
    gate_overlay = to_uint8(gate_rgb)
    mod_overlay = to_uint8(overlay_heatmap(input_np, mod_map))
    sup_overlay = to_uint8(overlay_heatmap(input_np, sup_map))

    # draw ROI box on full panels
    input_full = draw_roi_on_array(input_gt_np, roi)
    lem_full = draw_roi_on_array(lem_overlay, roi)
    gate_full = draw_roi_on_array(gate_overlay, roi)
    mod_full = draw_roi_on_array(mod_overlay, roi)
    sup_full = draw_roi_on_array(sup_overlay, roi)

    # crop ROI
    input_zoom = crop_roi(input_gt_np, roi)
    lem_zoom = crop_roi(lem_overlay, roi)
    gate_zoom = crop_roi(gate_overlay, roi)
    mod_zoom = crop_roi(mod_overlay, roi)
    sup_zoom = crop_roi(sup_overlay, roi)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(input_full).save(out_dir / "input_full.png")
    Image.fromarray(lem_full).save(out_dir / "P4_LEM_full.png")
    Image.fromarray(gate_full).save(out_dir / "P4_gate_full.png")
    Image.fromarray(mod_full).save(out_dir / "P4_modulated_full.png")
    Image.fromarray(sup_full).save(out_dir / "P4_suppressed_full.png")

    Image.fromarray(input_zoom).save(out_dir / "input_zoom.png")
    Image.fromarray(lem_zoom).save(out_dir / "P4_LEM_zoom.png")
    Image.fromarray(gate_zoom).save(out_dir / "P4_gate_zoom.png")
    Image.fromarray(mod_zoom).save(out_dir / "P4_modulated_zoom.png")
    Image.fromarray(sup_zoom).save(out_dir / "P4_suppressed_zoom.png")

    fig, axes = plt.subplots(2, 5, figsize=(16, 6.8))

    titles = [
        "Input",
        "LEM residual",
        "SCM gate (0.5-1.0)",
        "Modulated residual",
        "Attenuated residual",
    ]

    full_row = [input_full, lem_full, gate_full, mod_full, sup_full]
    zoom_row = [input_zoom, lem_zoom, gate_zoom, mod_zoom, sup_zoom]

    for j in range(5):
        axes[0, j].imshow(full_row[j])
        axes[0, j].axis("off")
        axes[0, j].set_title(titles[j], fontsize=13, pad=8)

        axes[1, j].imshow(zoom_row[j])
        axes[1, j].axis("off")

    axes[0, 0].text(
        -0.08, 0.5, "Full image",
        transform=axes[0, 0].transAxes,
        rotation=90, va="center", ha="right",
        fontsize=13, fontweight="bold"
    )
    axes[1, 0].text(
        -0.08, 0.5, "Zoomed ROI",
        transform=axes[1, 0].transAxes,
        rotation=90, va="center", ha="right",
        fontsize=13, fontweight="bold"
    )

    fig.tight_layout(pad=0.8, w_pad=0.5, h_pad=0.7)

    final_png = out_dir / "Fig_SCRM_mechanism_P4_focus.png"
    final_pdf = out_dir / "Fig_SCRM_mechanism_P4_focus.pdf"

    fig.savefig(final_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(final_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("Saved:", final_png)
    print("Saved:", final_pdf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--annotation", default=DEFAULT_ANN)
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(args)
