"""
Unified experiment entrypoint for TaylorSeer approximation-family optimization.

This replaces the original autoresearch training script for this local setup.
The runtime evaluation bucket is fixed to the interval-3 standing-baseline setup.
Edit the approximation hook in `taylor_utils/__init__.py`, then execute:

    /home/yjs/xdit_env/bin/python train.py
"""

# 中文概览：
# 这个脚本负责“单次实验”的完整闭环，而不是负责长期搜索调度。
# 一次运行的主流程是：
# 1. 校验路径和结果表是否存在
# 2. 清理旧采样输出
# 3. 调 sample.py 生成图片
# 4. 调 eval_image_diff.py 评估指标
# 5. 结合历史 results.tsv 判断本次状态
# 6. 归档 runs/<timestamp-...> 并向 results.tsv 追加一行

from __future__ import annotations

import argparse
import csv
import hashlib
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
PADE_TARGET_FILE = TAYLORSEER_DIR / "taylor_utils" / "__init__.py"

RESULT_FIELDS = [
    "timestamp",
    "commit",
    "branch",
    "pade_m",
    "pade_n",
    "max_order",
    "interval",
    "enable_pade",
    "pade_only_single_step",
    "pade_denom_threshold",
    "total_images",
    "batch_size",
    "seed",
    "cfg_scale",
    "num_sampling_steps",
    "lpips",
    "relative_l1",
    "ssim",
    "rmse",
    "psnr",
    "cosine",
    "approx_total_steps",
    "pade_steps",
    "taylor_steps",
    "mixed_steps",
    "pade_calls",
    "taylor_calls",
    "pade_step_ratio",
    "pade_call_ratio",
    "status",
    "comparison_bucket",
    "bucket_best",
    "paired_control_scope",
    "paired_control_timestamp",
    "paired_outcome",
    "paired_control_lpips",
    "paired_control_relative_l1",
    "paired_control_ssim",
    "paired_control_rmse",
    "paired_delta_lpips",
    "paired_delta_relative_l1",
    "paired_delta_ssim",
    "paired_delta_rmse",
    "latency_baseline_timestamp",
    "latency_baseline_sample_seconds",
    "sample_seconds_budget",
    "sample_seconds_ratio",
    "latency_within_budget",
    "pade_target_file",
    "pade_code_sha256",
    "pade_snapshot_file",
    "description",
    "sample_seconds",
    "eval_seconds",
    "total_seconds",
    "artifact_dir",
]

METRIC_TOLERANCE = 1e-6
STANDING_CONTROL_TIMESTAMP = "20260329-171615"
LATENCY_BUDGET_RATIO = 1.05


@dataclass
class ExperimentConfig:
    # Freeze runtime parameters at the interval-3 evaluation anchor.
    # The search surface is the approximation-family implementation behind
    # pade_formula_mn(), not the runtime parameters themselves.
    description: str = "standing-baseline approximation-family search at interval 3"
    pade_m: int = 1
    pade_n: int = 2
    max_order: int = 3
    interval: int = 3
    enable_pade: bool = True
    pade_only_single_step: bool = True
    pade_denom_threshold: float = 0.001
    total_images: int = 100
    batch_size: int = 2
    seed: int = 42
    cfg_scale: float = 1.5
    num_sampling_steps: int = 250
    timeout_seconds: int = 7200
    archive_images: bool = False


EXPERIMENT = ExperimentConfig()
DEFAULT_CONFIG = ExperimentConfig()
BOOL_FIELDS = {"enable_pade", "pade_only_single_step"}
INT_FIELDS = {
    "pade_m",
    "pade_n",
    "max_order",
    "interval",
    "total_images",
    "batch_size",
    "seed",
    "num_sampling_steps",
}
FLOAT_FIELDS = {"pade_denom_threshold", "cfg_scale"}
COMPARISON_FIELDS = [
    "pade_m",
    "pade_n",
    "max_order",
    "interval",
    "pade_only_single_step",
    "pade_denom_threshold",
    "total_images",
    "batch_size",
    "seed",
    "cfg_scale",
    "num_sampling_steps",
]
CONTROL_BASELINE_FIELDS = [
    "max_order",
    "interval",
    "total_images",
    "batch_size",
    "seed",
    "cfg_scale",
    "num_sampling_steps",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", type=str, default=None)
    parser.add_argument("--pade-m", type=int, default=None)
    parser.add_argument("--pade-n", type=int, default=None)
    parser.add_argument(
        "--enable-pade",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
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
    return parser.parse_args()


def merge_config(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig(**EXPERIMENT.__dict__)
    for arg_name, field_name in [
        ("description", "description"),
        ("pade_m", "pade_m"),
        ("pade_n", "pade_n"),
        ("enable_pade", "enable_pade"),
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

    if config.pade_m < 1 or config.pade_n < 1:
        raise ValueError("pade_m and pade_n must both be >= 1")
    if (config.pade_m, config.pade_n) not in {(1, 2), (2, 1)}:
        raise ValueError("current phase only allows Padé orders [1/2] or [2/1]")
    if config.max_order < (config.pade_m + config.pade_n):
        raise ValueError(
            f"max_order={config.max_order} must be at least pade_m + pade_n = "
            f"{config.pade_m + config.pade_n}"
        )
    if config.pade_denom_threshold <= 0:
        raise ValueError("pade_denom_threshold must be > 0")

    return config


def ensure_paths_exist() -> None:
    # 先把运行依赖的关键路径检查一遍，避免实验跑到中途才因路径缺失报错。
    required_paths = [
        XDIT_PYTHON,
        SAMPLE_SCRIPT,
        EVAL_SCRIPT,
        BASELINE_DIR,
        TAYLORSEER_DIR,
        PADE_TARGET_FILE,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))


def ensure_results_tsv() -> None:
    # results.tsv 是整个搜索过程的累计账本：
    # 不存在就创建；字段升级过就按当前表头重写，尽量兼容旧记录。
    if not RESULTS_TSV.exists():
        RESULTS_TSV.parent.mkdir(parents=True, exist_ok=True)
        with RESULTS_TSV.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t")
            writer.writeheader()
        return

    with RESULTS_TSV.open("r", newline="", encoding="utf-8") as handle:
        first_line = handle.readline().rstrip("\n")
        if first_line.split("\t") == RESULT_FIELDS:
            return
        handle.seek(0)
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)

    with RESULTS_TSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


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
    # 每轮实验前清理 pade_samples 下的关键产物，保证后续读取到的是本轮结果。
    SAMPLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.png", "eval.txt", "eval_metrics.json", "train_summary.json", "approx_stats.json"):
        for path in SAMPLE_OUTPUT_DIR.glob(pattern):
            if path.is_file():
                path.unlink()


def run_command(command: List[str], cwd: Path, timeout_seconds: int, label: str) -> float:
    # 统一封装外部命令执行，同时返回耗时，方便后面做 latency 对比。
    print(f"[train.py] Running {label}: {' '.join(command)}", flush=True)
    start = time.time()
    subprocess.run(command, cwd=cwd, check=True, timeout=timeout_seconds)
    return time.time() - start


def load_eval_metrics() -> Dict[str, float]:
    # 从评估脚本输出的 JSON 中提取最终聚合指标，供后续状态判定使用。
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


def load_approx_stats() -> Dict[str, float]:
    # 近似统计文件不是强依赖；缺失时返回零值，避免影响主流程落盘。
    stats_path = SAMPLE_OUTPUT_DIR / "approx_stats.json"
    if not stats_path.exists():
        return {
            "total_steps": 0,
            "pade_steps": 0,
            "taylor_steps": 0,
            "mixed_steps": 0,
            "pade_calls": 0,
            "taylor_calls": 0,
            "pade_step_ratio": 0.0,
            "pade_call_ratio": 0.0,
        }

    with stats_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return {
        "total_steps": int(payload.get("total_steps", 0)),
        "pade_steps": int(payload.get("pade_steps", 0)),
        "taylor_steps": int(payload.get("taylor_steps", 0)),
        "mixed_steps": int(payload.get("mixed_steps", 0)),
        "pade_calls": int(payload.get("pade_calls", 0)),
        "taylor_calls": int(payload.get("taylor_calls", 0)),
        "pade_step_ratio": float(payload.get("pade_step_ratio", 0.0)),
        "pade_call_ratio": float(payload.get("pade_call_ratio", 0.0)),
    }


def _format_bool(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value).strip().lower()


def _format_bucket_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _infer_enable_pade_from_legacy_row(row: Dict[str, str]) -> bool:
    explicit = row.get("enable_pade", "")
    if explicit:
        return _format_bool(explicit) == "true"

    pade_calls = row.get("pade_calls", "")
    taylor_calls = row.get("taylor_calls", "")
    if pade_calls and taylor_calls:
        try:
            if int(pade_calls) == 0 and int(taylor_calls) > 0:
                return False
        except ValueError:
            pass

    description = row.get("description", "").lower()
    if "pure taylor" in description:
        return False

    return True


def _row_field_value(row: Dict[str, str], field_name: str) -> object:
    if field_name == "enable_pade":
        return _infer_enable_pade_from_legacy_row(row)

    raw_value = row.get(field_name, "")
    if raw_value == "":
        return getattr(DEFAULT_CONFIG, field_name)

    if field_name in BOOL_FIELDS:
        return _format_bool(raw_value) == "true"
    if field_name in INT_FIELDS:
        return int(raw_value)
    if field_name in FLOAT_FIELDS:
        return float(raw_value)
    return raw_value


def _config_field_value(config: ExperimentConfig, field_name: str) -> object:
    return getattr(config, field_name)


def _comparison_bucket(config: ExperimentConfig) -> str:
    return ",".join(
        f"{field_name}={_format_bucket_value(_config_field_value(config, field_name))}"
        for field_name in COMPARISON_FIELDS
    )


def _row_matches_bucket(
    row: Dict[str, str],
    config: ExperimentConfig,
    *,
    include_enable_pade: bool,
) -> bool:
    fields = list(COMPARISON_FIELDS)
    if include_enable_pade:
        fields.append("enable_pade")
    return all(
        _row_field_value(row, field_name) == _config_field_value(config, field_name)
        for field_name in fields
    )


def _row_matches_control_baseline(row: Dict[str, str], config: ExperimentConfig) -> bool:
    return all(
        _row_field_value(row, field_name) == _config_field_value(config, field_name)
        for field_name in CONTROL_BASELINE_FIELDS
    )


def previous_completed_rows() -> Iterable[Dict[str, str]]:
    # 只把非 crash 的历史运行当作可比较样本。
    if not RESULTS_TSV.exists():
        return []
    with RESULTS_TSV.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [row for row in reader if row.get("status") != "crash"]


def row_metrics(row: Dict[str, str]) -> Dict[str, float]:
    return {
        "lpips": float(row["lpips"]),
        "relative_l1": float(row["relative_l1"]),
        "ssim": float(row["ssim"]),
        "rmse": float(row["rmse"]),
    }


def row_sample_seconds(row: Dict[str, str]) -> float:
    return float(row["sample_seconds"])


def _best_row(rows: Iterable[Dict[str, str]]) -> Optional[Dict[str, str]]:
    # 与 program.md 里的指标优先级保持一致：
    # LPIPS -> Relative L1 -> SSIM -> RMSE
    row_list = list(rows)
    if not row_list:
        return None
    return min(
        row_list,
        key=lambda row: (
            float(row["lpips"]),
            float(row["relative_l1"]),
            -float(row["ssim"]),
            float(row["rmse"]),
        ),
    )


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


def paired_control_summary(
    config: ExperimentConfig,
    metrics: Dict[str, float],
    sample_seconds: float,
    pade_code_sha256: str,
) -> Dict[str, object]:
    # 这里负责给当前候选找一个“纯 Taylor 对照组”。
    # 优先级是：
    # 1. 同代码快照下的 pure Taylor
    # 2. standing baseline
    # 3. 同 bucket 下历史上最合适的 pure Taylor
    default_summary: Dict[str, object] = {
        "paired_control_scope": "none",
        "paired_control_timestamp": "",
        "paired_outcome": "unpaired",
        "paired_control_lpips": "",
        "paired_control_relative_l1": "",
        "paired_control_ssim": "",
        "paired_control_rmse": "",
        "paired_delta_lpips": "",
        "paired_delta_relative_l1": "",
        "paired_delta_ssim": "",
        "paired_delta_rmse": "",
        "latency_baseline_timestamp": "",
        "latency_baseline_sample_seconds": "",
        "sample_seconds_budget": "",
        "sample_seconds_ratio": "",
        "latency_within_budget": "",
    }

    candidate_rows = [
        row
        for row in previous_completed_rows()
        if _row_matches_control_baseline(row, config)
        and not _row_field_value(row, "enable_pade")
    ]
    best_control = None
    control_scope = "none"
    same_code_controls = [
        row for row in candidate_rows if row.get("pade_code_sha256", "") == pade_code_sha256
    ]
    if same_code_controls:
        best_control = max(same_code_controls, key=lambda row: row.get("timestamp", ""))
        control_scope = "same_code_pure_taylor"
    else:
        standing_controls = [
            row
            for row in candidate_rows
            if row.get("timestamp", "") == STANDING_CONTROL_TIMESTAMP
        ]
        if standing_controls:
            best_control = standing_controls[0]
            control_scope = "standing_pure_taylor"
        elif candidate_rows:
            best_control = _best_row(candidate_rows)
            control_scope = "fallback_pure_taylor"

    if best_control is None:
        return default_summary

    control_metrics = row_metrics(best_control)
    control_sample_seconds = row_sample_seconds(best_control)
    sample_seconds_budget = control_sample_seconds * LATENCY_BUDGET_RATIO
    latency_within_budget = sample_seconds <= sample_seconds_budget

    if better_than(metrics, control_metrics):
        paired_outcome = "better_than_paired_control"
    elif better_than(control_metrics, metrics):
        paired_outcome = "worse_than_paired_control"
    else:
        paired_outcome = "tied_paired_control"

    return {
        "paired_control_scope": control_scope,
        "paired_control_timestamp": best_control.get("timestamp", ""),
        "paired_outcome": paired_outcome,
        "paired_control_lpips": control_metrics["lpips"],
        "paired_control_relative_l1": control_metrics["relative_l1"],
        "paired_control_ssim": control_metrics["ssim"],
        "paired_control_rmse": control_metrics["rmse"],
        "paired_delta_lpips": metrics["lpips"] - control_metrics["lpips"],
        "paired_delta_relative_l1": metrics["relative_l1"] - control_metrics["relative_l1"],
        "paired_delta_ssim": metrics["ssim"] - control_metrics["ssim"],
        "paired_delta_rmse": metrics["rmse"] - control_metrics["rmse"],
        "latency_baseline_timestamp": best_control.get("timestamp", ""),
        "latency_baseline_sample_seconds": control_sample_seconds,
        "sample_seconds_budget": sample_seconds_budget,
        "sample_seconds_ratio": sample_seconds / control_sample_seconds,
        "latency_within_budget": latency_within_budget,
    }


def choose_status(
    config: ExperimentConfig,
    metrics: Dict[str, float],
    sample_seconds: float,
    pade_code_sha256: str,
) -> Dict[str, object]:
    # 先和 pure Taylor 对照组比较，再结合延迟预算给当前候选打标签。
    # 这个状态会同时写入 results.tsv 和 runs/<...>/train_summary.json。
    paired = paired_control_summary(config, metrics, sample_seconds, pade_code_sha256)
    same_mode_rows = [
        row
        for row in previous_completed_rows()
        if _row_matches_bucket(row, config, include_enable_pade=True)
    ]
    best_same_mode = _best_row(same_mode_rows)
    is_mode_best = best_same_mode is None or better_than(metrics, row_metrics(best_same_mode))

    if config.enable_pade:
        latency_ok = paired["latency_within_budget"] is True
        if paired["paired_outcome"] == "better_than_paired_control" and latency_ok:
            status = "family_beats_baseline_within_latency_budget"
        elif paired["paired_outcome"] == "better_than_paired_control":
            status = "family_beats_baseline_but_too_slow"
        elif paired["paired_outcome"] == "tied_paired_control" and latency_ok:
            status = "family_ties_baseline_within_latency_budget"
        elif paired["paired_outcome"] == "tied_paired_control":
            status = "family_ties_baseline_but_too_slow"
        elif paired["paired_outcome"] == "worse_than_paired_control" and latency_ok:
            status = "family_within_latency_budget_but_loses_baseline"
        elif paired["paired_outcome"] == "worse_than_paired_control":
            status = "family_loses_baseline_and_is_too_slow"
        else:
            status = "family_no_baseline"
    else:
        status = "control_reference" if is_mode_best else "control_not_best_in_bucket"

    paired["status"] = status
    paired["comparison_bucket"] = _comparison_bucket(config)
    paired["bucket_best"] = is_mode_best
    return paired


def append_result(row: Dict[str, object]) -> None:
    with RESULTS_TSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t")
        writer.writerow(row)


def inspect_pade_target() -> Dict[str, object]:
    # 读取当前被搜索文件的源码和哈希，保证每次 run 都能追溯到具体实现版本。
    source_bytes = PADE_TARGET_FILE.read_bytes()
    return {
        "pade_target_file": str(PADE_TARGET_FILE),
        "pade_code_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_bytes": source_bytes,
    }


def archive_run(
    config: ExperimentConfig,
    timestamp: str,
    summary: Dict[str, object],
    pade_target: Dict[str, object],
) -> Path:
    # 每次运行都单独归档，保存：
    # - 当时的近似实现快照
    # - 评估结果
    # - 汇总 JSON
    # 这样后续可以脱离工作区直接复盘单次实验。
    run_dir = RUNS_DIR / f"{timestamp}-m{config.pade_m}-n{config.pade_n}-k{config.max_order}-i{config.interval}"
    run_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = run_dir / "pade_target_snapshot.py"
    snapshot_path.write_bytes(pade_target["source_bytes"])

    summary["artifact_dir"] = str(run_dir)
    summary["pade_snapshot_file"] = str(snapshot_path)
    with (run_dir / "train_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    for filename in ("eval.txt", "eval_metrics.json"):
        source = SAMPLE_OUTPUT_DIR / filename
        if source.exists():
            shutil.copy2(source, run_dir / filename)
    approx_stats = SAMPLE_OUTPUT_DIR / "approx_stats.json"
    if approx_stats.exists():
        shutil.copy2(approx_stats, run_dir / "approx_stats.json")

    if config.archive_images:
        image_dir = run_dir / "images"
        image_dir.mkdir(exist_ok=True)
        for image_path in SAMPLE_OUTPUT_DIR.glob("*.png"):
            shutil.copy2(image_path, image_dir / image_path.name)

    return run_dir


def build_sample_command(config: ExperimentConfig) -> List[str]:
    # 这里把固定 bucket 参数展开成 sample.py 的命令行参数。
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
        "--pade-denom-threshold",
        str(config.pade_denom_threshold),
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
    else:
        command.append("--no-enable-pade")
    if config.pade_only_single_step:
        command.append("--pade-only-single-step")
    else:
        command.append("--no-pade-only-single-step")
    return command


def build_eval_command() -> List[str]:
    # 评估脚本统一比较 baseline_samples 与 pade_samples。
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
        "comparison_bucket",
        "bucket_best",
        "pade_target_file",
        "pade_code_sha256",
        "pade_snapshot_file",
        "lpips",
        "relative_l1",
        "ssim",
        "rmse",
        "psnr",
        "cosine",
        "sample_seconds",
        "eval_seconds",
        "total_seconds",
        "latency_baseline_timestamp",
        "latency_baseline_sample_seconds",
        "sample_seconds_budget",
        "sample_seconds_ratio",
        "latency_within_budget",
        "pade_m",
        "pade_n",
        "max_order",
        "interval",
        "enable_pade",
        "pade_only_single_step",
        "pade_denom_threshold",
        "total_images",
        "batch_size",
        "seed",
        "cfg_scale",
        "num_sampling_steps",
        "paired_control_scope",
        "paired_control_timestamp",
        "paired_outcome",
        "paired_control_lpips",
        "paired_control_relative_l1",
        "paired_control_ssim",
        "paired_control_rmse",
        "paired_delta_lpips",
        "paired_delta_relative_l1",
        "paired_delta_ssim",
        "paired_delta_rmse",
        "approx_total_steps",
        "pade_steps",
        "taylor_steps",
        "mixed_steps",
        "pade_calls",
        "taylor_calls",
        "pade_step_ratio",
        "pade_call_ratio",
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
    # main() 负责把一次实验串起来：
    # 参数解析 -> 环境准备 -> 采样 -> 评估 -> 状态判定 -> 归档 -> 结果落表。
    args = parse_args()
    config = merge_config(args)

    ensure_paths_exist()
    ensure_results_tsv()
    cleanup_sample_output()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    commit = git_value("rev-parse", "--short", "HEAD")
    branch = git_value("branch", "--show-current")
    pade_target = inspect_pade_target()
    total_start = time.time()

    try:
        # 第一步：运行采样，实际调用 TaylorSeer-DiT/sample.py 生成样本。
        sample_seconds = run_command(
            build_sample_command(config),
            cwd=TAYLORSEER_DIR,
            timeout_seconds=config.timeout_seconds,
            label="sampling",
        )
        # 第二步：对生成结果做离线评估，产出 eval_metrics.json。
        eval_seconds = run_command(
            build_eval_command(),
            cwd=AUTORESEARCH_DIR,
            timeout_seconds=max(600, min(config.timeout_seconds, 3600)),
            label="evaluation",
        )
        # 第三步：读取指标和近似调用统计，结合历史结果给当前实验定性。
        metrics = load_eval_metrics()
        approx_stats = load_approx_stats()
        total_seconds = time.time() - total_start
        status_info = choose_status(
            config,
            metrics,
            sample_seconds,
            pade_target["pade_code_sha256"],
        )

        # 第四步：组装当前运行的完整摘要，既用于打印，也用于归档和写表。
        summary: Dict[str, object] = {
            "status": status_info["status"],
            "comparison_bucket": status_info["comparison_bucket"],
            "bucket_best": status_info["bucket_best"],
            "pade_target_file": pade_target["pade_target_file"],
            "pade_code_sha256": pade_target["pade_code_sha256"],
            "pade_snapshot_file": "",
            "lpips": metrics["lpips"],
            "relative_l1": metrics["relative_l1"],
            "ssim": metrics["ssim"],
            "rmse": metrics["rmse"],
            "psnr": metrics["psnr"],
            "cosine": metrics["cosine"],
            "sample_seconds": sample_seconds,
            "eval_seconds": eval_seconds,
            "total_seconds": total_seconds,
            "latency_baseline_timestamp": status_info["latency_baseline_timestamp"],
            "latency_baseline_sample_seconds": status_info["latency_baseline_sample_seconds"],
            "sample_seconds_budget": status_info["sample_seconds_budget"],
            "sample_seconds_ratio": status_info["sample_seconds_ratio"],
            "latency_within_budget": status_info["latency_within_budget"],
            "pade_m": config.pade_m,
            "pade_n": config.pade_n,
            "max_order": config.max_order,
            "interval": config.interval,
            "enable_pade": config.enable_pade,
            "pade_only_single_step": config.pade_only_single_step,
            "pade_denom_threshold": config.pade_denom_threshold,
            "total_images": config.total_images,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "cfg_scale": config.cfg_scale,
            "num_sampling_steps": config.num_sampling_steps,
            "paired_control_scope": status_info["paired_control_scope"],
            "paired_control_timestamp": status_info["paired_control_timestamp"],
            "paired_outcome": status_info["paired_outcome"],
            "paired_control_lpips": status_info["paired_control_lpips"],
            "paired_control_relative_l1": status_info["paired_control_relative_l1"],
            "paired_control_ssim": status_info["paired_control_ssim"],
            "paired_control_rmse": status_info["paired_control_rmse"],
            "paired_delta_lpips": status_info["paired_delta_lpips"],
            "paired_delta_relative_l1": status_info["paired_delta_relative_l1"],
            "paired_delta_ssim": status_info["paired_delta_ssim"],
            "paired_delta_rmse": status_info["paired_delta_rmse"],
            "approx_total_steps": approx_stats["total_steps"],
            "pade_steps": approx_stats["pade_steps"],
            "taylor_steps": approx_stats["taylor_steps"],
            "mixed_steps": approx_stats["mixed_steps"],
            "pade_calls": approx_stats["pade_calls"],
            "taylor_calls": approx_stats["taylor_calls"],
            "pade_step_ratio": approx_stats["pade_step_ratio"],
            "pade_call_ratio": approx_stats["pade_call_ratio"],
            "artifact_dir": "",
            "description": config.description,
        }
        # 第五步：将本轮产物写入 runs/<timestamp-...>/，形成可回溯快照。
        artifact_dir = archive_run(config, timestamp, summary, pade_target)

        # 第六步：把本轮结果追加到 results.tsv，供下一轮做历史比较。
        append_result(
            {
                "timestamp": timestamp,
                "commit": commit,
                "branch": branch,
                "pade_m": config.pade_m,
                "pade_n": config.pade_n,
                "max_order": config.max_order,
                "interval": config.interval,
                "enable_pade": str(config.enable_pade).lower(),
                "pade_only_single_step": str(config.pade_only_single_step).lower(),
                "pade_denom_threshold": f"{config.pade_denom_threshold:.6g}",
                "total_images": str(config.total_images),
                "batch_size": str(config.batch_size),
                "seed": str(config.seed),
                "cfg_scale": f"{config.cfg_scale:.6g}",
                "num_sampling_steps": str(config.num_sampling_steps),
                "lpips": f"{metrics['lpips']:.6f}",
                "relative_l1": f"{metrics['relative_l1']:.6f}",
                "ssim": f"{metrics['ssim']:.6f}",
                "rmse": f"{metrics['rmse']:.6f}",
                "psnr": f"{metrics['psnr']:.6f}",
                "cosine": f"{metrics['cosine']:.6f}",
                "approx_total_steps": str(approx_stats["total_steps"]),
                "pade_steps": str(approx_stats["pade_steps"]),
                "taylor_steps": str(approx_stats["taylor_steps"]),
                "mixed_steps": str(approx_stats["mixed_steps"]),
                "pade_calls": str(approx_stats["pade_calls"]),
                "taylor_calls": str(approx_stats["taylor_calls"]),
                "pade_step_ratio": f"{approx_stats['pade_step_ratio']:.6f}",
                "pade_call_ratio": f"{approx_stats['pade_call_ratio']:.6f}",
                "status": status_info["status"],
                "comparison_bucket": status_info["comparison_bucket"],
                "bucket_best": str(status_info["bucket_best"]).lower(),
                "paired_control_scope": status_info["paired_control_scope"],
                "paired_control_timestamp": status_info["paired_control_timestamp"],
                "paired_outcome": status_info["paired_outcome"],
                "paired_control_lpips": (
                    f"{status_info['paired_control_lpips']:.6f}"
                    if status_info["paired_control_lpips"] != ""
                    else ""
                ),
                "paired_control_relative_l1": (
                    f"{status_info['paired_control_relative_l1']:.6f}"
                    if status_info["paired_control_relative_l1"] != ""
                    else ""
                ),
                "paired_control_ssim": (
                    f"{status_info['paired_control_ssim']:.6f}"
                    if status_info["paired_control_ssim"] != ""
                    else ""
                ),
                "paired_control_rmse": (
                    f"{status_info['paired_control_rmse']:.6f}"
                    if status_info["paired_control_rmse"] != ""
                    else ""
                ),
                "paired_delta_lpips": (
                    f"{status_info['paired_delta_lpips']:.6f}"
                    if status_info["paired_delta_lpips"] != ""
                    else ""
                ),
                "paired_delta_relative_l1": (
                    f"{status_info['paired_delta_relative_l1']:.6f}"
                    if status_info["paired_delta_relative_l1"] != ""
                    else ""
                ),
                "paired_delta_ssim": (
                    f"{status_info['paired_delta_ssim']:.6f}"
                    if status_info["paired_delta_ssim"] != ""
                    else ""
                ),
                "paired_delta_rmse": (
                    f"{status_info['paired_delta_rmse']:.6f}"
                    if status_info["paired_delta_rmse"] != ""
                    else ""
                ),
                "latency_baseline_timestamp": status_info["latency_baseline_timestamp"],
                "latency_baseline_sample_seconds": (
                    f"{status_info['latency_baseline_sample_seconds']:.3f}"
                    if status_info["latency_baseline_sample_seconds"] != ""
                    else ""
                ),
                "sample_seconds_budget": (
                    f"{status_info['sample_seconds_budget']:.3f}"
                    if status_info["sample_seconds_budget"] != ""
                    else ""
                ),
                "sample_seconds_ratio": (
                    f"{status_info['sample_seconds_ratio']:.6f}"
                    if status_info["sample_seconds_ratio"] != ""
                    else ""
                ),
                "latency_within_budget": (
                    str(status_info["latency_within_budget"]).lower()
                    if status_info["latency_within_budget"] != ""
                    else ""
                ),
                "pade_target_file": pade_target["pade_target_file"],
                "pade_code_sha256": pade_target["pade_code_sha256"],
                "pade_snapshot_file": str(artifact_dir / "pade_target_snapshot.py"),
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
        # 即使失败也要把 crash 记录进 results.tsv，避免搜索历史出现“空洞”。
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
                "enable_pade": str(config.enable_pade).lower(),
                "pade_only_single_step": str(config.pade_only_single_step).lower(),
                "pade_denom_threshold": f"{config.pade_denom_threshold:.6g}",
                "total_images": str(config.total_images),
                "batch_size": str(config.batch_size),
                "seed": str(config.seed),
                "cfg_scale": f"{config.cfg_scale:.6g}",
                "num_sampling_steps": str(config.num_sampling_steps),
                "lpips": "0.000000",
                "relative_l1": "0.000000",
                "ssim": "0.000000",
                "rmse": "0.000000",
                "psnr": "0.000000",
                "cosine": "0.000000",
                "approx_total_steps": "0",
                "pade_steps": "0",
                "taylor_steps": "0",
                "mixed_steps": "0",
                "pade_calls": "0",
                "taylor_calls": "0",
                "pade_step_ratio": "0.000000",
                "pade_call_ratio": "0.000000",
                "status": "crash",
                "comparison_bucket": _comparison_bucket(config),
                "bucket_best": "false",
                "paired_control_scope": "",
                "paired_control_timestamp": "",
                "paired_outcome": "",
                "paired_control_lpips": "",
                "paired_control_relative_l1": "",
                "paired_control_ssim": "",
                "paired_control_rmse": "",
                "paired_delta_lpips": "",
                "paired_delta_relative_l1": "",
                "paired_delta_ssim": "",
                "paired_delta_rmse": "",
                "latency_baseline_timestamp": "",
                "latency_baseline_sample_seconds": "",
                "sample_seconds_budget": "",
                "sample_seconds_ratio": "",
                "latency_within_budget": "",
                "pade_target_file": pade_target["pade_target_file"],
                "pade_code_sha256": pade_target["pade_code_sha256"],
                "pade_snapshot_file": "",
                "description": f"{config.description} | {error_text}"[:200],
                "sample_seconds": "0.000",
                "eval_seconds": "0.000",
                "total_seconds": f"{total_seconds:.3f}",
                "artifact_dir": "",
            }
        )
        print("---", flush=True)
        print("status: crash", flush=True)
        print(f"pade_target_file: {pade_target['pade_target_file']}", flush=True)
        print(f"pade_code_sha256: {pade_target['pade_code_sha256']}", flush=True)
        print(f"error: {error_text}", flush=True)
        print(f"pade_m: {config.pade_m}", flush=True)
        print(f"pade_n: {config.pade_n}", flush=True)
        print(f"max_order: {config.max_order}", flush=True)
        print(f"interval: {config.interval}", flush=True)
        print(f"enable_pade: {config.enable_pade}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
