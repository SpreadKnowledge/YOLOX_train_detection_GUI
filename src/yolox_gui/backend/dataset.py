import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .logging import log_message


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class DatasetPreparation:
    dataset_dir: Path
    train_images: int
    val_images: int
    class_names: list[str]


def _log(log_callback, message: str):
    log_message(log_callback, message)


def _image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _label_for(image_path: Path, labels_dir: Path | None = None) -> Path | None:
    candidates = []
    if labels_dir is not None:
        try:
            relative = image_path.relative_to(image_path.parents[0])
        except ValueError:
            relative = Path(image_path.name)
        candidates.append(labels_dir / relative.with_suffix(".txt").name)
        candidates.append(labels_dir / image_path.with_suffix(".txt").name)
    candidates.append(image_path.with_suffix(".txt"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _collect_split(dataset_path: Path, split: str) -> list[tuple[Path, Path | None]]:
    images_dir = dataset_path / split / "images"
    labels_dir = dataset_path / split / "labels"
    return [(image, _label_for(image, labels_dir)) for image in _image_files(images_dir)]


def _collect_flat(dataset_path: Path) -> list[tuple[Path, Path | None]]:
    labels_dir = dataset_path / "labels"
    images = [
        path
        for path in _image_files(dataset_path)
        if "train" not in path.parts and "val" not in path.parts and "results" not in path.parts
    ]
    return [(image, _label_for(image, labels_dir)) for image in images]


def validate_yolo_dataset(dataset_path, class_names: list[str]) -> list[str]:
    dataset = Path(dataset_path)
    issues = []
    if not dataset.exists():
        return [f"Dataset path does not exist: {dataset}"]
    if not class_names:
        issues.append("At least one class name is required.")

    train_records = _collect_split(dataset, "train")
    val_records = _collect_split(dataset, "val")
    flat_records = _collect_flat(dataset)
    if not train_records and not flat_records:
        issues.append(
            "No images were found. Expected train/images and val/images, "
            "or a flat folder containing images and YOLO .txt labels."
        )
    if train_records and not val_records and len(train_records) < 2:
        issues.append("At least two training images are required when validation data is not provided.")
    if flat_records and len(flat_records) < 2:
        issues.append("At least two images are required for automatic train/val split.")
    return issues


def _split_records(dataset_path: Path) -> tuple[list[tuple[Path, Path | None]], list[tuple[Path, Path | None]]]:
    train_records = _collect_split(dataset_path, "train")
    val_records = _collect_split(dataset_path, "val")
    if train_records and val_records:
        return train_records, val_records

    records = train_records if train_records else _collect_flat(dataset_path)
    records = list(records)
    random.Random(0).shuffle(records)
    split_at = max(1, int(len(records) * 0.8))
    if split_at >= len(records):
        split_at = len(records) - 1
    return records[:split_at], records[split_at:]


def _safe_image_name(source: Path, used_names: set[str]) -> str:
    candidate = source.name
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate
    stem = source.stem
    suffix = source.suffix
    index = 1
    while True:
        candidate = f"{stem}_{index}{suffix}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        index += 1


def _parse_label_file(label_path: Path | None, width: int, height: int, class_count: int) -> list[tuple[int, float, float, float, float]]:
    if label_path is None or not label_path.exists():
        return []
    boxes = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Invalid label line {label_path}:{line_number}: {line}")
        class_id = int(float(parts[0]))
        if class_id < 0 or class_id >= class_count:
            raise ValueError(
                f"Class id {class_id} in {label_path}:{line_number} is outside 0..{class_count - 1}."
            )
        x_center, y_center, box_width, box_height = map(float, parts[1:5])
        x = (x_center - box_width / 2.0) * width
        y = (y_center - box_height / 2.0) * height
        w = box_width * width
        h = box_height * height
        x = max(0.0, min(float(width), x))
        y = max(0.0, min(float(height), y))
        w = max(0.0, min(float(width) - x, w))
        h = max(0.0, min(float(height) - y, h))
        if w > 0 and h > 0:
            boxes.append((class_id, x, y, w, h))
    return boxes


def _write_coco_split(records, output_dir: Path, split_name: str, class_names: list[str]) -> int:
    image_output_dir = output_dir / split_name
    image_output_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir = output_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    used_names = set()
    images = []
    annotations = []
    annotation_id = 1

    for image_id, (image_path, label_path) in enumerate(records, start=1):
        with Image.open(image_path) as image:
            width, height = image.size
        output_name = _safe_image_name(image_path, used_names)
        shutil.copy2(str(image_path), str(image_output_dir / output_name))

        images.append(
            {
                "id": image_id,
                "file_name": output_name,
                "width": width,
                "height": height,
            }
        )
        for class_id, x, y, w, h in _parse_label_file(label_path, width, height, len(class_names)):
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            annotation_id += 1

    categories = [{"id": index, "name": name} for index, name in enumerate(class_names)]
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    ann_name = "instances_train2017.json" if split_name == "train2017" else "instances_val2017.json"
    (annotations_dir / ann_name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(images)


def convert_yolo_to_coco(dataset_path, class_names: list[str], output_dir, log_callback=None) -> DatasetPreparation:
    dataset = Path(dataset_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    train_records, val_records = _split_records(dataset)
    if not train_records or not val_records:
        raise ValueError("Could not create both training and validation splits from the selected dataset.")

    train_count = _write_coco_split(train_records, output, "train2017", class_names)
    val_count = _write_coco_split(val_records, output, "val2017", class_names)
    (output / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    (output / "dataset_info.json").write_text(
        json.dumps(
            {
                "source": str(dataset),
                "format": "COCO converted from YOLO labels",
                "train_images": train_count,
                "val_images": val_count,
                "class_names": class_names,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _log(log_callback, f"prepared COCO dataset: {output}")
    _log(log_callback, f"train images: {train_count}")
    _log(log_callback, f"val images: {val_count}")
    return DatasetPreparation(output, train_count, val_count, class_names)


def prepare_dataset_for_yolox(dataset_path, class_names: list[str], output_root, project_name: str, log_callback=None) -> DatasetPreparation:
    issues = validate_yolo_dataset(dataset_path, class_names)
    if issues:
        raise ValueError("\n".join(issues))
    output_dir = Path(output_root).resolve() / project_name / "yolox_dataset"
    return convert_yolo_to_coco(dataset_path, class_names, output_dir, log_callback=log_callback)
