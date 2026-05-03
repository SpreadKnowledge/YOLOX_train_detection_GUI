"""Compatibility wrapper for the YOLOX training backend."""

import argparse
import json
import sys
from pathlib import Path

from src.yolox_gui.backend.dataset import prepare_dataset_for_yolox
from src.yolox_gui.backend.train import run_yolox_training


def create_yaml(project_name, train_data_path, class_names, save_directory):
    """Prepare a YOLOX COCO dataset and return its metadata path.

    The previous GUI created a YOLO data YAML file. YOLOX uses COCO JSON
    annotations through an exp file, so this function now prepares the COCO
    dataset and returns dataset_info.json for compatibility with older callers.
    """
    prepared = prepare_dataset_for_yolox(train_data_path, class_names, save_directory, project_name)
    return str(prepared.dataset_dir / "dataset_info.json")


def train_yolo(data_yaml, model_type, img_size, batch, epochs, model_save_path, project_name, train_data_path=None):
    """Backward-compatible name for YOLOX training."""
    if train_data_path is None:
        raise ValueError("train_data_path is required for YOLOX training.")
    info_path = Path(data_yaml)
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        class_names = info.get("class_names", [])
    else:
        raise FileNotFoundError(f"Dataset metadata was not found: {info_path}")
    return run_yolox_training(
        project_name=project_name,
        dataset_path=train_data_path,
        class_names=class_names,
        output_dir=model_save_path,
        model_size=model_type,
        img_size=img_size,
        epochs=epochs,
        batch_size=batch,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="YOLOX training compatibility wrapper")
    parser.add_argument("project_name")
    parser.add_argument("train_data_path")
    parser.add_argument("class_names")
    parser.add_argument("model_save_path")
    parser.add_argument("model_type")
    parser.add_argument("img_size", type=int)
    parser.add_argument("epochs", type=int)
    parser.add_argument("metadata_path")
    parser.add_argument("batch_size", type=int)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    class_names = [name.strip() for name in args.class_names.split(",") if name.strip()]
    return run_yolox_training(
        project_name=args.project_name,
        dataset_path=args.train_data_path,
        class_names=class_names,
        output_dir=args.model_save_path,
        model_size=args.model_type,
        img_size=args.img_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

