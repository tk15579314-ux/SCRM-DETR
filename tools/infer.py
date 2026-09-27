import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.cuda.amp import autocast

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.core import YAMLConfig


HIXRAY_CLASSES = [
    "Portable_Charger_1",
    "Portable_Charger_2",
    "Mobile_Phone",
    "Laptop",
    "Tablet",
    "Cosmetic",
    "Water",
    "Nonmetallic_Lighter",
]


def get_class_name(label_id):
    label_id = int(label_id)

    # 兼容 0-based 类别编号
    if 0 <= label_id < len(HIXRAY_CLASSES):
        return HIXRAY_CLASSES[label_id]

    # 兼容 1-based 类别编号
    if 1 <= label_id <= len(HIXRAY_CLASSES):
        return HIXRAY_CLASSES[label_id - 1]

    return f"class_{label_id}"


def postprocess(labels, boxes, scores, iou_threshold=0.55):
    def calculate_iou(box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2

        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)

        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)

        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area if union_area != 0 else 0
        return iou

    merged_labels = []
    merged_boxes = []
    merged_scores = []
    used_indices = set()

    if len(boxes) == 0:
        return [
            np.empty((0,), dtype=np.int64),
        ], [
            np.empty((0, 4), dtype=np.float32),
        ], [
            np.empty((0,), dtype=np.float32),
        ]

    for i in range(len(boxes)):
        if i in used_indices:
            continue

        current_box = boxes[i]
        current_label = labels[i]
        current_score = scores[i]

        boxes_to_merge = [current_box]
        scores_to_merge = [current_score]
        used_indices.add(i)

        for j in range(i + 1, len(boxes)):
            if j in used_indices:
                continue
            if labels[j] != current_label:
                continue

            other_box = boxes[j]
            iou = calculate_iou(current_box, other_box)

            if iou >= iou_threshold:
                boxes_to_merge.append(other_box)
                scores_to_merge.append(scores[j])
                used_indices.add(j)

        boxes_to_merge = np.asarray(boxes_to_merge)

        merged_box = [
            np.min(boxes_to_merge[:, 0]),
            np.min(boxes_to_merge[:, 1]),
            np.max(boxes_to_merge[:, 2]),
            np.max(boxes_to_merge[:, 3]),
        ]
        merged_score = max(scores_to_merge)

        merged_boxes.append(merged_box)
        merged_labels.append(current_label)
        merged_scores.append(merged_score)

    return [np.array(merged_labels)], [np.array(merged_boxes)], [np.array(merged_scores)]


def slice_image(image, slice_height, slice_width, overlap_ratio):
    if not 0 <= overlap_ratio < 1:
        raise ValueError(f"overlap_ratio must be in [0, 1), got {overlap_ratio}")

    img_width, img_height = image.size

    slices = []
    coordinates = []

    step_x = max(1, int(slice_width * (1 - overlap_ratio)))
    step_y = max(1, int(slice_height * (1 - overlap_ratio)))

    for y in range(0, img_height, step_y):
        for x in range(0, img_width, step_x):
            box = (
                x,
                y,
                min(x + slice_width, img_width),
                min(y + slice_height, img_height),
            )
            slice_img = image.crop(box)
            slices.append(slice_img)
            coordinates.append((x, y))

    return slices, coordinates


def merge_predictions(
    predictions,
    slice_coordinates,
    orig_image_size,
    slice_width,
    slice_height,
    threshold=0.30,
):
    merged_labels = []
    merged_boxes = []
    merged_scores = []

    orig_height, orig_width = orig_image_size

    for i, (label, boxes, scores) in enumerate(predictions):
        x_shift, y_shift = slice_coordinates[i]

        scores = np.array(scores).reshape(-1)
        valid_indices = scores > threshold

        valid_labels = np.array(label).reshape(-1)[valid_indices]
        valid_boxes = np.array(boxes).reshape(-1, 4)[valid_indices]
        valid_scores = scores[valid_indices]

        if valid_boxes.size == 0:
            continue

        valid_boxes = valid_boxes.copy()
        valid_boxes[:, [0, 2]] = np.clip(valid_boxes[:, [0, 2]] + x_shift, 0, orig_width)
        valid_boxes[:, [1, 3]] = np.clip(valid_boxes[:, [1, 3]] + y_shift, 0, orig_height)

        merged_labels.extend(valid_labels)
        merged_boxes.extend(valid_boxes)
        merged_scores.extend(valid_scores)

    return np.array(merged_labels), np.array(merged_boxes), np.array(merged_scores)


def draw(images, labels, boxes, scores, threshold=0.3, path=""):
    for i, im in enumerate(images):
        draw_obj = ImageDraw.Draw(im)

        scr = scores[i].detach().cpu()
        lab = labels[i].detach().cpu()
        box = boxes[i].detach().cpu()

        keep = scr > threshold
        lab = lab[keep]
        box = box[keep]
        scrs = scr[keep]

        print(f"Kept {len(scrs)} boxes with score > {threshold}")

        for j, b in enumerate(box):
            b = b.tolist()
            score = float(scrs[j])
            label_id = int(lab[j])
            class_name = get_class_name(label_id)

            draw_obj.rectangle(b, outline="red", width=2)
            draw_obj.text(
                (b[0], max(0, b[1] - 12)),
                text=f"{class_name}: {score:.2f}",
                font=ImageFont.load_default(),
                fill="blue",
            )

        if path == "":
            save_path = f"results_{i}.jpg"
        else:
            save_path = path

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        im.save(save_path)
        print(f"Saved visualization to: {save_path}")


def main(args):
    import torchvision.transforms as T

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is not available: {device}")

    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
    else:
        raise AttributeError("Only support resume to load model.state_dict by now.")

    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(device)
    model.eval()

    im_pil = Image.open(args.im_file).convert("RGB")
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]], device=device)

    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )

    im_data = transforms(im_pil)[None].to(device)

    if args.sliced:
        num_boxes = max(1, args.numberofboxes)

        aspect_ratio = w / h
        num_cols = max(1, int(np.sqrt(num_boxes * aspect_ratio)))
        num_rows = max(1, int(np.ceil(num_boxes / num_cols)))

        slice_height = max(1, h // num_rows)
        slice_width = max(1, w // num_cols)
        overlap_ratio = 0.2

        slices, coordinates = slice_image(
            im_pil,
            slice_height,
            slice_width,
            overlap_ratio,
        )

        predictions = []
        use_amp = device.type == "cuda"

        for slice_img in slices:
            slice_tensor = transforms(slice_img)[None].to(device)

            with torch.inference_mode():
                with autocast(enabled=use_amp):
                    output = model(
                        slice_tensor,
                        torch.tensor(
                            [[slice_img.size[0], slice_img.size[1]]]
                        ).to(device),
                    )

            labels, boxes, scores = output

            labels = labels.cpu().detach().numpy()
            boxes = boxes.cpu().detach().numpy()
            scores = scores.cpu().detach().numpy()

            predictions.append((labels, boxes, scores))

        merged_labels, merged_boxes, merged_scores = merge_predictions(
            predictions,
            coordinates,
            (h, w),
            slice_width,
            slice_height,
            threshold=args.threshold,
        )

        labels, boxes, scores = postprocess(
            merged_labels,
            merged_boxes,
            merged_scores,
        )

        labels = [torch.as_tensor(labels[0], dtype=torch.long, device=device)]
        boxes = [torch.as_tensor(boxes[0], dtype=torch.float32, device=device)]
        scores = [torch.as_tensor(scores[0], dtype=torch.float32, device=device)]

    else:
        with torch.inference_mode():
            output = model(im_data, orig_size)

        labels, boxes, scores = output

    image_name = os.path.splitext(os.path.basename(args.im_file))[0]
    output_path = os.path.join(args.output_dir, f"{image_name}_pred.jpg")

    draw(
        [im_pil],
        labels,
        boxes,
        scores,
        threshold=args.threshold,
        path=output_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str)
    parser.add_argument("-r", "--resume", type=str)
    parser.add_argument("-f", "--im-file", type=str)
    parser.add_argument("-s", "--sliced", action="store_true")
    parser.add_argument("-d", "--device", type=str, default="cpu")
    parser.add_argument("-nc", "--numberofboxes", type=int, default=25)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/rtdetr_r18vd_6x_hixray/vis_best",
    )

    args = parser.parse_args()
    main(args)
