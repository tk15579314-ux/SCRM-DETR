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
DEFAULT_OUT = "paper_eaai_records_20260912/figures/final_main_figures/scrm_mechanism"


def tensor_to_map(tensor, mode="abs_mean"):
    """
    tensor: [1, C, H, W]
    return: H x W numpy
    """
    t = tensor.detach().float()

    if mode == "abs_mean":
        m = t.abs().mean(dim=1)[0]
    elif mode == "mean":
        m = t.mean(dim=1)[0]
    else:
        raise ValueError(mode)

    return m.cpu().numpy()


def robust_norm(x, p_low=1.0, p_high=99.0):
    """
    Robust normalization for residual-related response maps.
    """
    x = np.asarray(x, dtype=np.float32)

    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)

    if hi <= lo:
        return np.zeros_like(x)

    x = (x - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0)


def resize_map(arr, size=(640, 640)):
    t = torch.from_numpy(arr).float()[None, None]
    t = F.interpolate(
        t,
        size=size,
        mode="bilinear",
        align_corners=False
    )
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


def draw_scaled_gt(image, boxes, orig_size):
    """
    Draw GT boxes after resizing original image to 640x640.
    """
    out = image.copy()
    draw = ImageDraw.Draw(out)

    ow, oh = orig_size
    sx = 640.0 / ow
    sy = 640.0 / oh

    for x1, y1, x2, y2 in boxes:
        box = [
            x1 * sx,
            y1 * sy,
            x2 * sx,
            y2 * sy,
        ]
        draw.rectangle(box, outline=(230, 30, 30), width=4)

    return out


def overlay_heatmap(image_np, response, cmap="turbo", alpha=0.48):
    response = np.clip(response, 0.0, 1.0)

    cm = plt.get_cmap(cmap)
    heat = cm(response)[..., :3]

    img = image_np.astype(np.float32) / 255.0
    blended = (1.0 - alpha) * img + alpha * heat

    return np.clip(blended, 0.0, 1.0)


def main(args):
    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("Loading configuration...")
    cfg = YAMLConfig(args.config, resume=args.checkpoint)

    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    # Follow tools/infer.py exactly: use EMA when available.
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
        print("Using EMA weights")
    else:
        state = checkpoint["model"]
        print("Using model weights")

    cfg.model.load_state_dict(state)

    model = cfg.model.to(device)
    model.eval()

    # --------------------------------------------------
    # Locate the two real SCXLCESoftBlock instances
    # --------------------------------------------------
    blocks = {}

    for name, module in model.named_modules():
        if module.__class__.__name__ == "SCXLCESoftBlock":
            scale_name = getattr(module, "scale_name", name)
            blocks[scale_name] = {
                "name": name,
                "module": module,
                "cache": {},
                "handles": [],
            }

    print("\n===== Found SCRM blocks =====")
    for k, v in blocks.items():
        m = v["module"]
        print(
            k,
            v["name"],
            "base=", m.sc_soft_base,
            "floor=", m.sc_gate_floor,
            "temp=", m.sc_gate_temperature,
            "alpha=", m.enhance_alpha,
        )

    if not blocks:
        raise RuntimeError("No SCXLCESoftBlock found.")

    # --------------------------------------------------
    # Hooks:
    # 1. block input x
    # 2. actual XLCE output
    # 3. actual consistency-gate logits
    # --------------------------------------------------
    def make_pre_hook(scale):
        def hook(module, inputs):
            blocks[scale]["cache"]["x"] = inputs[0].detach()
        return hook

    def make_xlce_hook(scale):
        def hook(module, inputs, output):
            blocks[scale]["cache"]["xlce_out"] = output.detach()
        return hook

    def make_logit_hook(scale):
        def hook(module, inputs, output):
            blocks[scale]["cache"]["gate_logit"] = output.detach()
        return hook

    for scale, info in blocks.items():
        m = info["module"]

        info["handles"].append(
            m.register_forward_pre_hook(make_pre_hook(scale))
        )

        info["handles"].append(
            m.xlce.register_forward_hook(make_xlce_hook(scale))
        )

        info["handles"].append(
            m.consistency_gate[0].register_forward_hook(
                make_logit_hook(scale)
            )
        )

    # --------------------------------------------------
    # Image preprocessing: same as tools/infer.py
    # --------------------------------------------------
    image_path = Path(args.image)
    im_orig = Image.open(image_path).convert("RGB")
    orig_size = im_orig.size

    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    im_data = transform(im_orig)[None].to(device)

    print("\nRunning one forward pass...")
    with torch.no_grad():
        _ = model(im_data)

    # Remove hooks.
    for info in blocks.values():
        for h in info["handles"]:
            h.remove()

    # --------------------------------------------------
    # Input image for visualization
    # --------------------------------------------------
    input_640 = im_orig.resize((640, 640), Image.Resampling.BILINEAR)

    boxes = load_gt_boxes(
        args.annotation,
        image_path.name,
    )

    input_gt = draw_scaled_gt(
        input_640,
        boxes,
        orig_size,
    )

    input_np = np.asarray(input_640)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_gt.save(out_dir / "input_gt.png")

    results = {}

    # --------------------------------------------------
    # Compute the three quantities from the ACTUAL forward
    # --------------------------------------------------
    preferred_order = ["P4", "P3"]

    for scale in preferred_order:
        if scale not in blocks:
            continue

        m = blocks[scale]["module"]
        cache = blocks[scale]["cache"]

        required = ["x", "xlce_out", "gate_logit"]
        for r in required:
            if r not in cache:
                raise RuntimeError(
                    f"{scale}: missing captured tensor {r}"
                )

        x = cache["x"]
        xlce_out = cache["xlce_out"]
        gate_logit = cache["gate_logit"]

        residual = xlce_out - x

        gate = torch.sigmoid(
            gate_logit / m.sc_gate_temperature
        )

        gate_soft = (
            m.sc_soft_base
            + (1.0 - m.sc_soft_base) * gate
        )

        gate_soft = (
            m.sc_gate_floor
            + (1.0 - m.sc_gate_floor) * gate_soft
        )

        modulated = (
            m.enhance_alpha
            * gate_soft
            * residual
        )

        print(f"\n===== {scale} tensors =====")
        print("x          :", tuple(x.shape))
        print("residual   :", tuple(residual.shape))
        print("gate_soft  :", tuple(gate_soft.shape))
        print("modulated  :", tuple(modulated.shape))
        print(
            "gate range :",
            float(gate_soft.min()),
            float(gate_soft.max()),
            float(gate_soft.mean()),
        )

        # LEM: mean absolute local enhancement residual.
        lem_map_raw = tensor_to_map(
            residual,
            mode="abs_mean"
        )

        # SCM: mean soft gate.
        gate_map_raw = tensor_to_map(
            gate_soft,
            mode="mean"
        )

        # SCRM: mean absolute gated residual.
        mod_map_raw = tensor_to_map(
            modulated,
            mode="abs_mean"
        )

        lem_map = resize_map(
            robust_norm(lem_map_raw),
            (640, 640)
        )

        # IMPORTANT:
        # Final model has soft_base=0.5 and floor=0.
        # Visualize gate on the physical fixed [0.5, 1.0] scale,
        # NOT per-image min-max normalization.
        gate_min = m.sc_gate_floor + (
            1.0 - m.sc_gate_floor
        ) * m.sc_soft_base

        gate_max = 1.0

        gate_map = (
            gate_map_raw - gate_min
        ) / max(gate_max - gate_min, 1e-8)

        gate_map = np.clip(gate_map, 0.0, 1.0)
        gate_map = resize_map(
            gate_map,
            (640, 640)
        )

        mod_map = resize_map(
            robust_norm(mod_map_raw),
            (640, 640)
        )

        lem_overlay = overlay_heatmap(
            input_np, lem_map
        )

        gate_overlay = overlay_heatmap(
            input_np, gate_map
        )

        mod_overlay = overlay_heatmap(
            input_np, mod_map
        )

        plt.imsave(
            out_dir / f"{scale}_LEM_residual.png",
            lem_overlay
        )

        plt.imsave(
            out_dir / f"{scale}_SCM_gate.png",
            gate_overlay
        )

        plt.imsave(
            out_dir / f"{scale}_SCRM_modulated.png",
            mod_overlay
        )

        # Also save pure heatmaps for inspection.
        plt.imsave(
            out_dir / f"{scale}_LEM_residual_heatmap.png",
            lem_map,
            cmap="turbo",
            vmin=0,
            vmax=1,
        )

        plt.imsave(
            out_dir / f"{scale}_SCM_gate_heatmap.png",
            gate_map,
            cmap="turbo",
            vmin=0,
            vmax=1,
        )

        plt.imsave(
            out_dir / f"{scale}_SCRM_modulated_heatmap.png",
            mod_map,
            cmap="turbo",
            vmin=0,
            vmax=1,
        )

        results[scale] = {
            "lem": lem_overlay,
            "gate": gate_overlay,
            "mod": mod_overlay,
            "gate_mean": float(gate_soft.mean()),
            "gate_min": float(gate_soft.min()),
            "gate_max": float(gate_soft.max()),
        }

    # --------------------------------------------------
    # Paper composite: 2 rows x 4 columns
    # --------------------------------------------------
    scales = [
        s for s in ["P4", "P3"]
        if s in results
    ]

    fig, axes = plt.subplots(
        len(scales),
        4,
        figsize=(14, 7 if len(scales) == 2 else 3.8),
    )

    if len(scales) == 1:
        axes = np.expand_dims(axes, axis=0)

    column_titles = [
        "Input",
        "LEM residual",
        "SCM gate",
        "Modulated residual",
    ]

    input_gt_np = np.asarray(input_gt)

    for row, scale in enumerate(scales):
        panels = [
            input_gt_np,
            results[scale]["lem"],
            results[scale]["gate"],
            results[scale]["mod"],
        ]

        for col in range(4):
            axes[row, col].imshow(panels[col])
            axes[row, col].axis("off")

            if row == 0:
                axes[row, col].set_title(
                    column_titles[col],
                    fontsize=13,
                    pad=8,
                )

        axes[row, 0].text(
            -0.05,
            0.5,
            scale,
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=14,
            fontweight="bold",
        )

    fig.tight_layout(
        pad=0.8,
        w_pad=0.5,
        h_pad=0.8,
    )

    final_png = out_dir / "Fig_SCRM_mechanism_visualization.png"
    final_pdf = out_dir / "Fig_SCRM_mechanism_visualization.pdf"

    fig.savefig(
        final_png,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        final_pdf,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print("\n===== DONE =====")
    print("Output directory:", out_dir)
    print("Final PNG:", final_png)
    print("Final PDF:", final_pdf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
    )

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CKPT,
    )

    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
    )

    parser.add_argument(
        "--annotation",
        default=DEFAULT_ANN,
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUT,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    args = parser.parse_args()
    main(args)
