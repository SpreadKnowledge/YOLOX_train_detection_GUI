"""Simple precision/recall calculation for YOLO-label datasets with a YOLOX checkpoint."""

import csv
import os
import random
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

from src.yolox_gui.backend.infer import YoloxPredictor


DATASET_DIR = "PATH/TO/YOUR/DIRECTORY"
MODEL_PATH = "PATH/TO/YOUR/MODEL.pth"
MODEL_SIZE = "yolox_s"
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
IMG_SIZE = 640


def create_output_dirs():
    base_dir = Path(DATASET_DIR) / "test_results"
    detect_dir = base_dir / "detection_images"
    base_dir.mkdir(exist_ok=True)
    detect_dir.mkdir(exist_ok=True)
    return base_dir, detect_dir


def generate_colors(num_classes: int) -> Dict[int, Tuple[int, int, int]]:
    random.seed(42)
    return {
        index: (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        for index in range(num_classes)
    }


def calculate_iou(box1, box2):
    b1_x1, b1_y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    b1_x2, b1_y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    b2_x1, b2_y1 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    b2_x2, b2_y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    return inter_area / (b1_area + b2_area - inter_area)


def evaluate_detection(gt_boxes, gt_classes, pred_boxes, pred_classes, iou_threshold=0.5):
    if len(pred_boxes) == 0:
        return (1.0, 1.0, 1.0) if len(gt_boxes) == 0 else (0.0, 0.0, 0.0)
    if len(gt_boxes) == 0:
        return 0.0, 0.0, 0.0

    true_positives = 0
    used_gt = set()
    for pred_idx, pred in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1
        for index, gt in enumerate(gt_boxes):
            if index in used_gt:
                continue
            iou = calculate_iou(pred, gt)
            if iou > best_iou and pred_classes[pred_idx] == gt_classes[index]:
                best_iou = iou
                best_gt_idx = index
        if best_iou >= iou_threshold:
            true_positives += 1
            used_gt.add(best_gt_idx)

    precision = true_positives / len(pred_boxes)
    recall = true_positives / len(gt_boxes)
    f_value = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f_value


def _bbox_xyxy_to_yolo_normalized(bbox, image_width, image_height):
    x1, y1, x2, y2 = bbox
    width = max(x2 - x1, 0.0)
    height = max(y2 - y1, 0.0)
    x_center = x1 + width / 2
    y_center = y1 + height / 2
    return [x_center / image_width, y_center / image_height, width / image_width, height / image_height]


def main():
    predictor = YoloxPredictor(
        checkpoint_path=MODEL_PATH,
        model_size=MODEL_SIZE,
        img_size=IMG_SIZE,
        conf_threshold=CONF_THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
    )
    class_names = predictor.class_names
    num_classes = len(class_names)
    class_colors = generate_colors(num_classes)

    base_dir, detect_dir = create_output_dirs()
    image_files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    detection_counts = []
    total_precision = 0
    total_recall = 0
    total_f_value = 0
    evaluated_images = 0

    for img_file in image_files:
        img_path = Path(DATASET_DIR) / img_file
        label_path = img_path.with_suffix(".txt")
        if not label_path.exists():
            continue

        gt_boxes = []
        gt_classes = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            class_id, x, y, w, h = map(float, line.strip().split()[:5])
            gt_boxes.append([x, y, w, h])
            gt_classes.append(int(class_id))

        image = cv2.imread(str(img_path))
        if image is None:
            continue
        annotated, detections = predictor.predict_frame(image)
        pred_boxes = []
        pred_classes = []
        class_counts = {index: 0 for index in range(num_classes)}

        for label, confidence, bbox in detections:
            class_id = class_names.index(label) if label in class_names else 0
            class_counts[class_id] += 1
            pred_boxes.append(_bbox_xyxy_to_yolo_normalized(bbox, image.shape[1], image.shape[0]))
            pred_classes.append(class_id)
            color = class_colors[class_id]
            x1, y1, x2, y2 = [int(value) for value in bbox]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        detection_counts.append([img_file] + [class_counts[index] for index in range(num_classes)])
        precision, recall, f_value = evaluate_detection(gt_boxes, gt_classes, pred_boxes, pred_classes)
        total_precision += precision
        total_recall += recall
        total_f_value += f_value
        evaluated_images += 1

        cv2.imwrite(str(detect_dir / f"{img_path.stem}_det.jpg"), annotated)

    detection_csv_path = base_dir / "num_of_detections.csv"
    with detection_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["filename"] + list(class_names))
        writer.writerows(detection_counts)

    divisor = max(evaluated_images, 1)
    txt_path = base_dir / "precision_recall_f-value.txt"
    txt_path.write_text(
        "\n".join(
            [
                f"Average Precision: {total_precision / divisor:.4f}",
                f"Average Recall: {total_recall / divisor:.4f}",
                f"Average F-value: {total_f_value / divisor:.4f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

