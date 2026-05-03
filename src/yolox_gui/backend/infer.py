import mimetypes
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import torch

from .paths import (
    add_yolox_to_syspath,
    find_class_names,
    infer_checkpoint_num_classes,
    resolve_exp_file,
    validate_model_size,
)


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VALID_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


def _log(callback, message: str):
    if callback:
        callback(message)
    else:
        print(message, flush=True)


def is_valid_image(file_path) -> bool:
    path = Path(file_path)
    if path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
        return False
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type is None or mime_type.startswith("image/")


def is_valid_video(file_path) -> bool:
    path = Path(file_path)
    if path.suffix.lower() not in VALID_VIDEO_EXTENSIONS:
        return False
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type is None or mime_type.startswith("video/")


def get_media_files(directory) -> tuple[list[Path], list[Path]]:
    root = Path(directory).resolve()
    image_files = []
    video_files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if is_valid_image(path):
            image_files.append(path)
        elif is_valid_video(path):
            video_files.append(path)
    return sorted(image_files), sorted(video_files)


class YoloxPredictor:
    def __init__(
        self,
        *,
        checkpoint_path,
        model_size: str,
        img_size: int = 640,
        conf_threshold: float = 0.3,
        nms_threshold: float = 0.45,
        device: str = "auto",
        class_names: Iterable[str] | None = None,
    ):
        validate_model_size(model_size)
        add_yolox_to_syspath()

        from yolox.data.data_augment import ValTransform
        from yolox.exp import get_exp
        from yolox.utils import postprocess, vis

        self.postprocess = postprocess
        self.vis = vis
        self.preproc = ValTransform(legacy=False)
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.model_size = model_size
        self.num_classes = infer_checkpoint_num_classes(self.checkpoint_path)
        self.class_names = list(class_names) if class_names else find_class_names(self.checkpoint_path, self.num_classes)
        self.class_names = tuple(self.class_names[: self.num_classes])

        exp = get_exp(str(resolve_exp_file(model_size)), None)
        exp.num_classes = self.num_classes
        exp.test_conf = conf_threshold
        exp.nmsthre = nms_threshold
        exp.test_size = (int(img_size), int(img_size))
        self.exp = exp
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        if device == "auto":
            device = "gpu" if torch.cuda.is_available() else "cpu"
        if device == "gpu" and not torch.cuda.is_available():
            raise RuntimeError("GPU was requested, but PyTorch CUDA is not available.")
        self.device = device

        model = exp.get_model()
        model.eval()
        ckpt = torch.load(str(self.checkpoint_path), map_location="cpu")
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=False)
        if self.device == "gpu":
            model.cuda()
        self.model = model

    def predict_frame(self, frame):
        height, width = frame.shape[:2]
        ratio = min(self.exp.test_size[0] / height, self.exp.test_size[1] / width)
        image, _ = self.preproc(frame, None, self.exp.test_size)
        image = torch.from_numpy(image).unsqueeze(0).float()
        if self.device == "gpu":
            image = image.cuda()

        with torch.no_grad():
            outputs = self.model(image)
            outputs = self.postprocess(
                outputs,
                self.num_classes,
                self.conf_threshold,
                self.nms_threshold,
                class_agnostic=True,
            )

        output = outputs[0]
        if output is None:
            return frame.copy(), []

        output = output.cpu()
        bboxes = output[:, 0:4] / ratio
        classes = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        detections = []
        for bbox, cls, score in zip(bboxes, classes, scores):
            class_id = int(cls)
            label = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            detections.append((label, float(score), [float(value) for value in bbox.tolist()]))
        visual = self.vis(frame, bboxes, scores, classes, self.conf_threshold, self.class_names)
        return visual, detections

    def predict_image_file(self, image_path):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        return self.predict_frame(image)


def _write_detection_txt(path: Path, detections):
    with path.open("w", encoding="utf-8") as file:
        for label, score, bbox in detections:
            x1, y1, x2, y2 = bbox
            file.write(f"{label} {score:.4f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")


def _process_video(video_path: Path, predictor: YoloxPredictor, output_dir: Path, progress_callback=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    video_dir = output_dir / "videos" / video_path.stem
    video_dir.mkdir(parents=True, exist_ok=True)
    output_video = video_dir / f"{video_path.stem}_yolox.mp4"
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_index = 0
    detection_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        annotated, detections = predictor.predict_frame(frame)
        writer.write(annotated)
        if detections:
            image_path = video_dir / f"frame_{detection_index:04d}.jpg"
            txt_path = video_dir / f"frame_{detection_index:04d}.txt"
            cv2.imwrite(str(image_path), annotated)
            _write_detection_txt(txt_path, detections)
            detection_index += 1
        frame_index += 1
        if progress_callback and frame_index % 30 == 0:
            if total_frames:
                progress_callback(f"video {video_path.name}: {frame_index}/{total_frames} frames")
            else:
                progress_callback(f"video {video_path.name}: {frame_index} frames")

    cap.release()
    writer.release()
    _log(progress_callback, f"video saved: {output_video}")


def detect_images(
    images_folder,
    checkpoint_path,
    *,
    model_size: str,
    callback=None,
    progress_callback=None,
    conf_threshold: float = 0.3,
    nms_threshold: float = 0.45,
    img_size: int = 640,
    output_dir=None,
):
    source = Path(images_folder).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input folder was not found: {source}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint was not found: {checkpoint}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(output_dir).resolve() if output_dir else source / "results" / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    _log(progress_callback, f"selected model size: {model_size}")
    _log(progress_callback, f"checkpoint path: {checkpoint}")
    _log(progress_callback, f"output: {results_dir}")
    _log(progress_callback, f"confidence threshold: {conf_threshold}")
    _log(progress_callback, f"NMS threshold: {nms_threshold}")

    image_files, video_files = get_media_files(source)
    if not image_files and not video_files:
        raise ValueError(f"No valid image or video files were found in: {source}")

    predictor = YoloxPredictor(
        checkpoint_path=checkpoint,
        model_size=model_size,
        img_size=img_size,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
    )
    _log(progress_callback, f"num classes: {predictor.num_classes}")
    _log(progress_callback, f"device: {predictor.device}")

    image_output_dir = results_dir / "images"
    image_output_dir.mkdir(parents=True, exist_ok=True)
    for index, image_path in enumerate(image_files, start=1):
        if progress_callback:
            progress_callback(f"image {index}/{len(image_files)}: {image_path.name}")
        annotated, detections = predictor.predict_image_file(image_path)
        output_image = image_output_dir / image_path.name
        output_txt = image_output_dir / f"{image_path.stem}.txt"
        cv2.imwrite(str(output_image), annotated)
        _write_detection_txt(output_txt, detections)

    for index, video_path in enumerate(video_files, start=1):
        _log(progress_callback, f"video {index}/{len(video_files)}: {video_path.name}")
        _process_video(video_path, predictor, results_dir, progress_callback=progress_callback)

    if callback:
        callback(str(results_dir))
    return results_dir

