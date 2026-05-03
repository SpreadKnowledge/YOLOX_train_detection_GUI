import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

from src.system_report import write_environment_report

from .dataset import prepare_dataset_for_yolox
from .logging import log_message
from .paths import (
    create_runtime_exp_file,
    ensure_yolox_root,
    resolve_exp_file,
    validate_model_size,
    yolox_subprocess_env,
)
from .weights import ensure_yolox_weight


YOLOX_PROGRESS_RE = re.compile(r"epoch:\s*(?P<epoch>\d+)/(?P<total>\d+)")
YOLOX_TRAINING_DONE_MARKER = "Training of experiment is done"
YOLOX_FATAL_MARKERS = (
    " | ERROR    | ",
    "Traceback (most recent call last):",
    "UnicodeDecodeError:",
)


def _log(log_callback, message: str):
    log_message(log_callback, message)


def _is_yolox_fatal_line(line: str) -> bool:
    return any(marker in line for marker in YOLOX_FATAL_MARKERS)


def _device_summary() -> tuple[str, bool]:
    try:
        import torch

        if torch.cuda.is_available():
            return f"GUI_DEVICE device=cuda torch_cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}", True
        return f"GUI_DEVICE device=cpu torch_cuda={torch.version.cuda} gpu=not_available", False
    except Exception as exc:
        return f"GUI_DEVICE device=unknown error={exc}", False


def build_train_command(exp_file: Path, weight_path: Path, batch_size: int) -> list[str]:
    yolox_root = ensure_yolox_root()
    return [
        sys.executable,
        str(yolox_root / "tools" / "train.py"),
        "-f",
        str(exp_file),
        "-d",
        "1",
        "-b",
        str(batch_size),
        "-c",
        str(weight_path),
        "--logger",
        "tensorboard",
    ]


def run_yolox_training(
    *,
    project_name: str,
    dataset_path,
    class_names: list[str],
    output_dir,
    model_size: str,
    img_size: int,
    epochs: int,
    batch_size: int,
    log_callback=None,
) -> int:
    validate_model_size(model_size)
    output_root = Path(output_dir).resolve()
    run_dir = output_root / project_name
    run_dir.mkdir(parents=True, exist_ok=True)

    _log(log_callback, f"selected model size: {model_size}")
    exp_source = resolve_exp_file(model_size)
    _log(log_callback, f"YOLOX base exp file: {exp_source}")
    _log(log_callback, f"dataset path: {Path(dataset_path).resolve()}")
    _log(log_callback, f"output directory: {run_dir}")

    device_line, has_cuda = _device_summary()
    _log(log_callback, device_line)
    weight_path = ensure_yolox_weight(model_size, log_callback=log_callback)
    if not has_cuda:
        _log(
            log_callback,
            "YOLOX official training backend requires CUDA. "
            "Install a CUDA-enabled PyTorch build and run on a GPU for training.",
        )
        return 2

    dataset = prepare_dataset_for_yolox(
        dataset_path,
        class_names,
        output_root,
        project_name,
        log_callback=log_callback,
    )
    exp_file = create_runtime_exp_file(
        model_size=model_size,
        output_dir=output_root,
        exp_name=project_name,
        num_classes=len(class_names),
        img_size=img_size,
        target_path=run_dir / f"{project_name}_yolox_exp.py",
        dataset_dir=dataset.dataset_dir,
        epochs=epochs,
    )
    _log(log_callback, f"runtime exp file: {exp_file}")

    command = build_train_command(exp_file, weight_path, batch_size)
    _log(log_callback, "command: " + " ".join(f'"{part}"' if " " in part else part for part in command))

    start_time = time.time()
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
    fatal_error_seen = False
    training_completed_seen = False
    for raw_line in iter(process.stdout.readline, ""):
        line = raw_line.rstrip()
        if not line:
            continue
        _log(log_callback, line)
        if YOLOX_TRAINING_DONE_MARKER in line:
            training_completed_seen = True
        elif _is_yolox_fatal_line(line):
            fatal_error_seen = True
        match = YOLOX_PROGRESS_RE.search(line)
        if match:
            epoch = int(match.group("epoch"))
            total = max(int(match.group("total")), 1)
            elapsed = max(time.time() - start_time, 0.1)
            eta = max(total - epoch, 0) * (elapsed / max(epoch, 1))
            _log(log_callback, f"GUI_PROGRESS epoch={epoch} total={total} elapsed={elapsed:.1f} eta={eta:.1f}")

    return_code = process.wait()
    if return_code == 0 and fatal_error_seen and not training_completed_seen:
        return_code = 1
        _log(log_callback, "YOLOX emitted a fatal error while returning exit code 0; treating the run as failed.")
    elif return_code == 0 and fatal_error_seen and training_completed_seen:
        _log(log_callback, "YOLOX training completion was confirmed; keeping exit code 0 despite earlier warning-like output.")
    _log(log_callback, f"YOLOX training process exited with code {return_code}")

    classes_path = run_dir / "classes.txt"
    classes_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")
    if return_code == 0:
        report_path = write_environment_report(run_dir)
        _log(log_callback, f"GUI_ARTIFACT environment={report_path}")
        for candidate in ("best_ckpt.pth", "latest_ckpt.pth"):
            weight_candidate = run_dir / candidate
            if weight_candidate.exists():
                _log(log_callback, f"GUI_ARTIFACT weights={weight_candidate}")
                break
    return return_code


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="YOLOX GUI training backend")
    parser.add_argument("--project-name", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--class-names", required=True, help="Comma-separated class names")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-size", required=True)
    parser.add_argument("--img-size", required=True, type=int)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    class_names = [name.strip() for name in args.class_names.split(",") if name.strip()]
    return run_yolox_training(
        project_name=args.project_name,
        dataset_path=args.dataset_path,
        class_names=class_names,
        output_dir=args.output_dir,
        model_size=args.model_size,
        img_size=args.img_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    raise SystemExit(main())
