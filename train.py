"""
Unified experiment entrypoint for TaylorSeer Padé parameter search.

This replaces the original autoresearch training script for this local setup.
Edit the EXPERIMENT block below between runs, then execute:

    /home/yjs/xdit_env/bin/python train.py
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


AUTORESEARCH_DIR = Path("/home/yjs/autoresearch")
XDIT_PYTHON = Path("/home/yjs/xdit_env/bin/python")
TAYLORSEER_DIR = Path("/home/yjs/TaylorSeer/TaylorSeer-DiT")
SAMPLE_SCRIPT = TAYLORSEER_DIR / "sample.py"
EVAL_SCRIPT = Path("/home/yjs/eval_image_diff.py")
BASELINE_DIR = Path("/home/yjs/baseline_samples")
SAMPLE_OUTPUT_DIR = TAYLORSEER_DIR / "pade_samples"
RUNS_DIR = AUTORESEARCH_DIR / "runs"
RESULTS_TSV = AUTORESEARCH_DIR / "results.tsv"

RESULT_FIELDS = [
    "timestamp",
    "commit",
    "branch",
    "pade_m",
    "pade_n",
    "max_order",
    "interval",
    "lpips",
    "relative_l1",
    "ssim",
    "rmse",
    "psnr",
    "cosine",
    "status",
    "description",
    "sample_seconds",
    "eval_seconds",
    "total_seconds",
    "artifact_dir",
]

METRIC_TOLERANCE = 1e-6


@dataclass
class ExperimentConfig:
    # Edit this block between experiments.
    description: str = "local search [2/1] pade"
    pade_m: int = 2
    pade_n: int = 1
    max_order: int = 3
    interval: int = 4
    enable_pade: bool = True
    total_images: int = 200
    batch_size: int = 2
    seed: int = 42
    cfg_scale: float = 1.5
    num_sampling_steps: int = 250
    timeout_seconds: int = 7200
    archive_images: bool = False


EXPERIMENT = ExperimentConfig()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", type=str, default=None)
    parser.add_argument("--pade-m", type=int, default=None)
    parser.add_argument("--pade-n", type=int, default=None)
    parser.add_argument("--max-order", type=int, default=None)
    parser.add_argument("--interval", type=int, default=None)
    parser.add_argument("--total-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--num-sampling-steps", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument(
        "--archive-images",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--disable-pade", action="store_true")
    return parser.parse_args()


def merge_config(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig(**EXPERIMENT.__dict__)
    for arg_name, field_name in [
        ("description", "description"),
        ("pade_m", "pade_m"),
        ("pade_n", "pade_n"),
        ("max_order", "max_order"),
        ("interval", "interval"),
        ("total_images", "total_images"),
        ("batch_size", "batch_size"),
        ("seed", "seed"),
        ("cfg_scale", "cfg_scale"),
        ("num_sampling_steps", "num_sampling_steps"),
        ("timeout_seconds", "timeout_seconds"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            setattr(config, field_name, value)

    if args.archive_images is not None:
        config.archive_images = args.archive_images
    if args.disable_pade:
        config.enable_pade = False

    if config.pade_m < 1 or config.pade_n < 1:
        raise ValueError("pade_m and pade_n must both be >= 1")
    if config.max_order < (config.pade_m + config.pade_n):
        raise ValueError(
            f"max_order={config.max_order} must be at least pade_m + pade_n = "
            f"{config.pade_m + config.pade_n}"
        )

    return config


def ensure_paths_exist() -> None:
    required_paths = [
        XDIT_PYTHON,
        SAMPLE_SCRIPT,
        EVAL_SCRIPT,
        BASELINE_DIR,
        TAYLORSEER_DIR,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))


def ensure_results_tsv() -> None:
    if RESULTS_TSV.exists():
        return
    RESULTS_TSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_TSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t")
        writer.writeheader()


def git_value(*args: str) -> str:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(AUTORESEARCH_DIR), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return output or "unknown"
    except subprocess.SubprocessError:
        return "unknown"


def cleanup_sample_output() -> None:
    SAMPLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.png", "eval.txt", "eval_metrics.json", "train_summary.json"):
        for path in SAMPLE_OUTPUT_DIR.glob(pattern):
            if path.is_file():
                path.unlink()


def run_command(command: List[str], cwd: Path, timeout_seconds: int, label: str) -> float:
    print(f"[train.py] Running {label}: {' '.join(command)}", flush=True)
    start = time.time()
    subprocess.run(command, cwd=cwd, check=True, timeout=timeout_seconds)
    return time.time() - start


def load_eval_metrics() -> Dict[str, float]:
    metrics_path = SAMPLE_OUTPUT_DIR / "eval_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing evaluation output: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    overall = payload.get("overall_average")
    if not isinstance(overall, dict):
        raise ValueError("eval_metrics.json does not contain overall_average")

    return {
        "lpips": float(overall["LPIPS"]),
        "relative_l1": float(overall["Relative L1"]),
        "ssim": float(overall["SSIM"]),
        "rmse": float(overall["RMSE"]),
        "psnr": float(overall["PSNR"]),
        "cosine": float(overall["Cosine Similarity"]),
    }


def previous_success_rows() -> Iterable[Dict[str, str]]:
    if not RESULTS_TSV.exists():
        return []
    with RESULTS_TSV.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [row for row in reader if row.get("status") == "keep"]


def better_than(current: Dict[str, float], previous: Dict[str, float]) -> bool:
    comparisons = [
        ("lpips", "min"),
        ("relative_l1", "min"),
        ("ssim", "max"),
        ("rmse", "min"),
    ]
    for key, direction in comparisons:
        cur = current[key]
        prev = previous[key]
        if abs(cur - prev) <= METRIC_TOLERANCE:
            continue
        if direction == "min":
            return cur < prev
        return cur > prev
    return False


def choose_status(metrics: Dict[str, float]) -> str:
    best_rows = list(previous_success_rows())
    if not best_rows:
        return "keep"

    def row_metrics(row: Dict[str, str]) -> Dict[str, float]:
        return {
            "lpips": float(row["lpips"]),
            "relative_l1": float(row["relative_l1"]),
            "ssim": float(row["ssim"]),
            "rmse": float(row["rmse"]),
        }

    best_previous = min(
        best_rows,
        key=lambda row: (
            float(row["lpips"]),
            float(row["relative_l1"]),
            -float(row["ssim"]),
            float(row["rmse"]),
        ),
    )
    return "keep" if better_than(metrics, row_metrics(best_previous)) else "discard"


def append_result(row: Dict[str, object]) -> None:
    with RESULTS_TSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t")
        writer.writerow(row)


def archive_run(
    config: ExperimentConfig,
    timestamp: str,
    summary: Dict[str, object],
) -> Path:
    run_dir = RUNS_DIR / f"{timestamp}-m{config.pade_m}-n{config.pade_n}-k{config.max_order}-i{config.interval}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary["artifact_dir"] = str(run_dir)
    with (run_dir / "train_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    for filename in ("eval.txt", "eval_metrics.json"):
        source = SAMPLE_OUTPUT_DIR / filename
        if source.exists():
            shutil.copy2(source, run_dir / filename)

    if config.archive_images:
        image_dir = run_dir / "images"
        image_dir.mkdir(exist_ok=True)
        for image_path in SAMPLE_OUTPUT_DIR.glob("*.png"):
            shutil.copy2(image_path, image_dir / image_path.name)

    return run_dir


def build_sample_command(config: ExperimentConfig) -> List[str]:
    command = [
        str(XDIT_PYTHON),
        str(SAMPLE_SCRIPT),
        "--pade-m",
        str(config.pade_m),
        "--pade-n",
        str(config.pade_n),
        "--max-order",
        str(config.max_order),
        "--interval",
        str(config.interval),
        "--total-images",
        str(config.total_images),
        "--batch-size",
        str(config.batch_size),
        "--seed",
        str(config.seed),
        "--cfg-scale",
        str(config.cfg_scale),
        "--num-sampling-steps",
        str(config.num_sampling_steps),
    ]
    if config.enable_pade:
        command.append("--enable-pade")
    return command


def build_eval_command() -> List[str]:
    return [
        str(XDIT_PYTHON),
        str(EVAL_SCRIPT),
        "--path1",
        str(BASELINE_DIR),
        "--path2",
        str(SAMPLE_OUTPUT_DIR),
        "--output-dir",
        str(SAMPLE_OUTPUT_DIR),
    ]


def summary_lines(summary: Dict[str, object]) -> List[str]:
    ordered_keys = [
        "status",
        "lpips",
        "relative_l1",
        "ssim",
        "rmse",
        "psnr",
        "cosine",
        "sample_seconds",
        "eval_seconds",
        "total_seconds",
        "pade_m",
        "pade_n",
        "max_order",
        "interval",
        "artifact_dir",
        "description",
    ]
    lines = ["---"]
    for key in ordered_keys:
        value = summary[key]
        if isinstance(value, float):
            lines.append(f"{key}: {value:.6f}")
        else:
            lines.append(f"{key}: {value}")
    return lines


def main() -> int:
    args = parse_args()
    config = merge_config(args)

    ensure_paths_exist()
    ensure_results_tsv()
    cleanup_sample_output()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    commit = git_value("rev-parse", "--short", "HEAD")
    branch = git_value("branch", "--show-current")
    total_start = time.time()

    try:
        sample_seconds = run_command(
            build_sample_command(config),
            cwd=TAYLORSEER_DIR,
            timeout_seconds=config.timeout_seconds,
            label="sampling",
        )
        eval_seconds = run_command(
            build_eval_command(),
            cwd=AUTORESEARCH_DIR,
            timeout_seconds=max(600, min(config.timeout_seconds, 3600)),
            label="evaluation",
        )
        metrics = load_eval_metrics()
        total_seconds = time.time() - total_start
        status = choose_status(metrics)

        summary: Dict[str, object] = {
            "status": status,
            "lpips": metrics["lpips"],
            "relative_l1": metrics["relative_l1"],
            "ssim": metrics["ssim"],
            "rmse": metrics["rmse"],
            "psnr": metrics["psnr"],
            "cosine": metrics["cosine"],
            "sample_seconds": sample_seconds,
            "eval_seconds": eval_seconds,
            "total_seconds": total_seconds,
            "pade_m": config.pade_m,
            "pade_n": config.pade_n,
            "max_order": config.max_order,
            "interval": config.interval,
            "artifact_dir": "",
            "description": config.description,
        }
        artifact_dir = archive_run(config, timestamp, summary)

        append_result(
            {
                "timestamp": timestamp,
                "commit": commit,
                "branch": branch,
                "pade_m": config.pade_m,
                "pade_n": config.pade_n,
                "max_order": config.max_order,
                "interval": config.interval,
                "lpips": f"{metrics['lpips']:.6f}",
                "relative_l1": f"{metrics['relative_l1']:.6f}",
                "ssim": f"{metrics['ssim']:.6f}",
                "rmse": f"{metrics['rmse']:.6f}",
                "psnr": f"{metrics['psnr']:.6f}",
                "cosine": f"{metrics['cosine']:.6f}",
                "status": status,
                "description": config.description,
                "sample_seconds": f"{sample_seconds:.3f}",
                "eval_seconds": f"{eval_seconds:.3f}",
                "total_seconds": f"{total_seconds:.3f}",
                "artifact_dir": str(artifact_dir),
            }
        )

        for line in summary_lines(summary):
            print(line, flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        total_seconds = time.time() - total_start
        error_text = f"{type(exc).__name__}: {exc}"
        print(error_text, file=sys.stderr, flush=True)
        traceback.print_exc()
        append_result(
            {
                "timestamp": timestamp,
                "commit": commit,
                "branch": branch,
                "pade_m": config.pade_m,
                "pade_n": config.pade_n,
                "max_order": config.max_order,
                "interval": config.interval,
                "lpips": "0.000000",
                "relative_l1": "0.000000",
                "ssim": "0.000000",
                "rmse": "0.000000",
                "psnr": "0.000000",
                "cosine": "0.000000",
                "status": "crash",
                "description": f"{config.description} | {error_text}"[:200],
                "sample_seconds": "0.000",
                "eval_seconds": "0.000",
                "total_seconds": f"{total_seconds:.3f}",
                "artifact_dir": "",
            }
        )
        print("---", flush=True)
        print("status: crash", flush=True)
        print(f"error: {error_text}", flush=True)
        print(f"pade_m: {config.pade_m}", flush=True)
        print(f"pade_n: {config.pade_n}", flush=True)
        print(f"max_order: {config.max_order}", flush=True)
        print(f"interval: {config.interval}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
