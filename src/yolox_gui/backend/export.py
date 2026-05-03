import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from .logging import log_message
from .paths import (
    create_runtime_exp_file,
    ensure_yolox_root,
    infer_checkpoint_num_classes,
    validate_model_size,
    yolox_subprocess_env,
)


def _log(log_callback, message: str):
    log_message(log_callback, message)


def build_onnx_export_command(
    *,
    exp_file: Path,
    checkpoint_path: Path,
    output_path: Path,
    opset: int,
    simplify: bool,
    dynamic_axes: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(ensure_yolox_root() / "tools" / "export_onnx.py"),
        "-f",
        str(exp_file),
        "-c",
        str(checkpoint_path),
        "--output-name",
        str(output_path),
        "-o",
        str(opset),
    ]
    if dynamic_axes:
        command.append("--dynamic")
    if not simplify:
        command.append("--no-onnxsim")
    return command


def check_onnx_runtime(onnx_path: Path, img_size: int, log_callback=None):
    try:
        import onnxruntime as ort
    except Exception as exc:
        _log(log_callback, f"ONNX Runtime check skipped: onnxruntime is not importable ({exc})")
        return

    try:
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_meta = session.get_inputs()[0]
        dummy = np.random.rand(1, 3, int(img_size), int(img_size)).astype(np.float32)
        outputs = session.run(None, {input_meta.name: dummy})
    except Exception as exc:
        _log(log_callback, f"ONNX Runtime check failed: {exc}")
        return
    shapes = [tuple(output.shape) for output in outputs if hasattr(output, "shape")]
    _log(log_callback, f"ONNX Runtime check completed: output shapes={shapes}")


def export_onnx(
    *,
    checkpoint_path,
    model_size: str,
    output_path,
    img_size: int = 640,
    opset: int = 11,
    simplify: bool = False,
    dynamic_axes: bool = False,
    log_callback=None,
) -> Path:
    validate_model_size(model_size)
    checkpoint = Path(checkpoint_path).resolve()
    output = Path(output_path).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint was not found: {checkpoint}")
    output.parent.mkdir(parents=True, exist_ok=True)

    _log(log_callback, f"selected model size: {model_size}")
    _log(log_callback, f"checkpoint path: {checkpoint}")
    _log(log_callback, f"output ONNX path: {output}")
    _log(log_callback, f"input image size: {img_size}")
    _log(log_callback, f"opset version: {opset}")
    _log(log_callback, f"simplify: {simplify}")
    _log(log_callback, f"dynamic axes: {dynamic_axes}")

    num_classes = infer_checkpoint_num_classes(checkpoint)
    exp_file = create_runtime_exp_file(
        model_size=model_size,
        output_dir=output.parent,
        exp_name=f"{model_size}_export",
        num_classes=num_classes,
        img_size=img_size,
        target_path=output.parent / f"{model_size}_export_exp.py",
    )
    _log(log_callback, f"runtime exp file: {exp_file}")

    command = build_onnx_export_command(
        exp_file=exp_file,
        checkpoint_path=checkpoint,
        output_path=output,
        opset=opset,
        simplify=simplify,
        dynamic_axes=dynamic_axes,
    )
    _log(log_callback, "command: " + " ".join(f'"{part}"' if " " in part else part for part in command))

    process = subprocess.Popen(
        command,
        cwd=str(ensure_yolox_root()),
        env=yolox_subprocess_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for raw_line in iter(process.stdout.readline, ""):
        line = raw_line.rstrip()
        if line:
            _log(log_callback, line)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ONNX export failed with exit code {return_code}")
    if not output.exists():
        raise FileNotFoundError(f"ONNX export finished, but output was not found: {output}")

    _log(log_callback, "ONNX export completed")
    check_onnx_runtime(output, img_size, log_callback=log_callback)
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="YOLOX GUI ONNX export backend")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--model-size", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--opset", type=int, default=11)
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--dynamic-axes", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    export_onnx(
        checkpoint_path=args.checkpoint_path,
        model_size=args.model_size,
        output_path=args.output_path,
        img_size=args.img_size,
        opset=args.opset,
        simplify=args.simplify,
        dynamic_axes=args.dynamic_axes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
