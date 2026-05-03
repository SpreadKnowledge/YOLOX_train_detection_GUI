"""Compatibility wrapper for YOLOX image and video inference."""

from src.yolox_gui.backend.infer import (
    VALID_IMAGE_EXTENSIONS,
    VALID_VIDEO_EXTENSIONS,
    detect_images,
    get_media_files,
    is_valid_image,
    is_valid_video,
)


__all__ = [
    "VALID_IMAGE_EXTENSIONS",
    "VALID_VIDEO_EXTENSIONS",
    "detect_images",
    "get_media_files",
    "is_valid_image",
    "is_valid_video",
]

