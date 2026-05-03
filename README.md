# YOLOX-train-detection-GUI

This is a desktop GUI application dedicated to YOLOX training, inference, and model export.

<img width="2559" height="1522" alt="スクリーンショット 2026-05-03 201508" src="https://github.com/user-attachments/assets/7d7d0b25-86d8-465b-9d37-20cc19b1442c" />

## Overview

YOLOX-train-detection-GUI is a Python desktop application for running YOLOX workflows from a CustomTkinter GUI. It keeps the existing training, inference, camera, log, file selection, XML conversion, metrics, and system report structure while replacing the model backend with YOLOX.

The current implementation vendors selected YOLOX code under `src/yolox_gui/vendor/yolox/`. The original `third_party/YOLOX` checkout is still present as source reference and can be removed later after release packaging is finalized.

## Features

- YOLOX model size selection for training and inference.
- Automatic YOLOX base model weight check and download into `weights/yolox/`.
- YOLO-label dataset preparation with COCO JSON conversion for YOLOX training.
- YOLOX training through the official YOLOX training tool.
- Image, folder, video, and camera inference through a YOLOX backend wrapper.
- Model format conversion subwindow with ONNX export.
- ONNX Runtime smoke check after export when `onnxruntime` is installed.
- Existing utility modules for XML-to-TXT conversion, metrics calculation, and system reports.

## Supported YOLOX Model Sizes

- `yolox_nano`
- `yolox_tiny`
- `yolox_s`
- `yolox_m`
- `yolox_l`
- `yolox_x`

## Installation

Python 3.10 or newer is recommended.

```bash
git clone <this-repository-url>
cd YOLOX-train-detection-GUI
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

The default `requirements.txt` uses CUDA 12.4 PyTorch wheels. If your environment needs CPU-only or a different CUDA version, adjust the PyTorch index and versions before installing.

## How To Run

```bash
python main.py
```

The GUI opens with the training view. Use the sidebar for training, image/video inference, camera inference, and model export.

## Training Workflow

1. Select a project name.
2. Select a dataset folder.
3. Select an output folder.
4. Select a YOLOX model size.
5. Enter input size, epochs, batch size, and class names.
6. Start training and monitor the log panel.

The backend checks:

- selected model size
- YOLOX exp file
- local base weight path
- dataset path
- output directory

YOLOX official training currently expects CUDA. The GUI logs a clear error instead of crashing when CUDA is unavailable.

## Dataset Format Notes

The GUI accepts a YOLO-label dataset and prepares a COCO-style dataset for YOLOX.

Preferred layout:

```text
dataset/
  train/images/
  train/labels/
  val/images/
  val/labels/
```

A flat folder with image files and matching `.txt` labels is also supported. In that case the backend creates an 80/20 train/validation split and writes a prepared COCO dataset under the selected output folder.

YOLO labels must use:

```text
class_id x_center y_center width height
```

with normalized coordinates.

## Inference Workflow

1. Select an image/video folder.
2. Select a YOLOX checkpoint (`.pth` or `.pt`).
3. Select the matching YOLOX model size.
4. Set confidence and NMS thresholds.
5. Start inference.

Results are saved under:

```text
<input-folder>/results/<timestamp>/
```

Camera inference uses the same YOLOX checkpoint and model size selection, and saves captured frames plus detection text files to the selected save folder.

## Model Export / ONNX Export

Open `Model Export` from the sidebar.

Implemented:

- ONNX export
- input checkpoint path
- model size
- output ONNX path
- input image size
- opset version
- simplify option
- dynamic axes option
- ONNX Runtime smoke check when available

Planned but not implemented yet:

- TensorRT
- OpenVINO
- ncnn

Those controls are intentionally disabled or marked as TODO.

## Base Model Auto-Download

YOLOX base weights are managed in:

```text
weights/yolox/
```

The model size to URL table is centralized in:

```text
src/yolox_gui/backend/weights.py
```

When a selected base weight is missing, the backend logs:

- selected model size
- local weight path
- weight exists / not found
- download started
- download completed
- download failed

Weight URLs are taken from the YOLOX model zoo information in the YOLOX repository.

## License

This project is licensed under the Apache License 2.0. See `LICENSE`.

## Third-Party Components

This project uses selected YOLOX components as third-party code.

YOLOX is licensed under the Apache License 2.0.

The YOLOX vendor copy includes an origin notice:

```text
src/yolox_gui/vendor/yolox/NOTICE_ORIGIN.md
```

Additional tracking notes are in:

```text
NOTICE
THIRD_PARTY_LICENSES/
```

## TODO / Roadmap

- Finalize whether `third_party/YOLOX` remains as a submodule/reference or is removed after validating the vendor copy.
- Add a packaged dependency lock file for reproducible installation.
- Add richer dataset validation reports in the GUI.
- Add class-name selection for inference when using checkpoints from external training runs.
- Add TensorRT, OpenVINO, and ncnn export workflows.
- Add automated GUI smoke tests for model selection and export windows.

