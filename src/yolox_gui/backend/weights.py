from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

from .logging import log_message
from .paths import MODEL_SIZES, validate_model_size, weights_root


YOLOX_WEIGHT_URLS = {
    "yolox_nano": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth",
    "yolox_tiny": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth",
    "yolox_s": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth",
    "yolox_m": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth",
    "yolox_l": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth",
    "yolox_x": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth",
}


def get_weight_path(model_size: str) -> Path:
    validate_model_size(model_size)
    filename = f"{model_size}.pth"
    return weights_root() / filename


def ensure_yolox_weight(model_size: str, log_callback=None) -> Path:
    validate_model_size(model_size)

    def log(message: str):
        log_message(log_callback, message)

    weight_path = get_weight_path(model_size)
    log(f"selected model size: {model_size}")
    log(f"local weight path: {weight_path}")

    if weight_path.exists():
        log("weight exists")
        return weight_path

    log("weight not found")
    url = YOLOX_WEIGHT_URLS.get(model_size)
    if not url:
        raise RuntimeError(
            f"TODO: pretrained weight URL is not configured for {model_size}. "
            "Check the YOLOX model zoo and update src/yolox_gui/backend/weights.py."
        )

    weight_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = weight_path.with_suffix(weight_path.suffix + ".download")
    log(f"download started: {url}")
    try:
        urlretrieve(url, tmp_path)
        tmp_path.replace(weight_path)
    except (OSError, URLError) as exc:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        log(f"download failed: {exc}")
        raise RuntimeError(f"Failed to download YOLOX base weight for {model_size}: {exc}") from exc

    log("download completed")
    return weight_path


__all__ = ["MODEL_SIZES", "YOLOX_WEIGHT_URLS", "ensure_yolox_weight", "get_weight_path"]
