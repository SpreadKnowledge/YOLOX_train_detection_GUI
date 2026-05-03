import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
from PIL import Image, ImageTk

from src.yolox_gui.backend.infer import YoloxPredictor


def normalize_path(path):
    if not path:
        return path
    return str(Path(path).resolve())


class CameraDetection:
    def __init__(self, model_path, model_size="yolox_s", conf_threshold=0.5, nms_threshold=0.45, img_size=640):
        self.predictor = YoloxPredictor(
            checkpoint_path=model_path,
            model_size=model_size,
            img_size=img_size,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
        )
        self.conf_threshold = conf_threshold
        self.cap = None
        self.running = False
        self.save_dir = ""
        self.scene_id = 0
        self.last_frame = None
        self.last_annotated = None
        self.last_detections = []

    def start_camera(self, camera_id):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError("Unable to open camera")
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def set_save_directory(self, directory):
        self.save_dir = directory

    def show_camera_stream(self, display_label):
        self.running = True
        threading.Thread(target=self._update_stream, args=(display_label,), daemon=True).start()

    def _update_stream(self, display_label):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            annotated, detections = self.predictor.predict_frame(frame)
            self.last_frame = frame.copy()
            self.last_annotated = annotated.copy()
            self.last_detections = detections

            image = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            image = self._resize_image_to_fit(image, display_label.winfo_width(), display_label.winfo_height())
            pil_image = Image.fromarray(image)

            try:
                display_label.after(0, self._display_frame, display_label, pil_image)
            except Exception:
                break

            time.sleep(0.03)

    def _display_frame(self, display_label, pil_image):
        if not self.running:
            return
        try:
            photo = ImageTk.PhotoImage(image=pil_image)
            display_label.configure(image=photo, text="")
            display_label.image = photo
        except Exception:
            self.running = False

    def _resize_image_to_fit(self, image, max_width, max_height):
        height, width = image.shape[:2]
        max_width = max(int(max_width), 1)
        max_height = max(int(max_height), 1)
        aspect_ratio = width / height

        if aspect_ratio > max_width / max_height:
            new_width = max_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(new_height * aspect_ratio)

        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def capture_frame(self):
        if not self.save_dir:
            return None

        frame = self.last_frame
        annotated = self.last_annotated
        detections = self.last_detections
        if frame is None or annotated is None:
            if not self.cap:
                return None
            ret, frame = self.cap.read()
            if not ret:
                return None
            annotated, detections = self.predictor.predict_frame(frame)

        self.scene_id += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        base_filename = f"{timestamp}_{self.scene_id:04d}"

        save_dir_path = Path(self.save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)

        origin_image_path = str(save_dir_path / f"{base_filename}_origin.png")
        detection_image_path = str(save_dir_path / f"{base_filename}_detection.jpg")
        txt_path = str(save_dir_path / f"{base_filename}_detection.txt")

        cv2.imwrite(origin_image_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        cv2.imwrite(detection_image_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(txt_path, "w", encoding="utf-8") as file:
            for label, confidence, bbox in detections:
                x1, y1, x2, y2 = bbox
                file.write(f"{label} {confidence:.4f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")

        return origin_image_path, detection_image_path, txt_path

    def stop(self):
        self.running = False
        self.stop_camera()

