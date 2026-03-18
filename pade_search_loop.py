#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path("/home/yjs/autoresearch")
TRAIN_PY = ROOT / "train.py"
RESULTS_TSV = ROOT / "results.tsv"
RUNS_DIR = ROOT / "runs"
RUN_LOG = ROOT / "run.log"
PYTHON = Path("/home/yjs/xdit_env/bin/python")


@dataclass(frozen=True)
class Candidate:
    description: str
    pade_m: int
    pade_n: int
    max_order: int
    interval: int = 4
    enable_pade: bool = True
    pade_only_single_step: bool = True
    pade_denom_threshold: float = 1e-4

    @property
    def key(self) -> tuple[int, int, int, int, bool, str]:
        return (
            self.pade_m,
            self.pade_n,
            self.max_order,
            self.interval,
            self.pade_only_single_step,
            format(self.pade_denom_threshold, ".6g"),
        )


def read_results() -> list[dict[str, str]]:
    if not RESULTS_TSV.exists():
        return []
    with RESULTS_TSV.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader)


def iter_keep_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("status") == "keep"]


def metric_key(row: dict[str, str]) -> tuple[float, float, float, float]:
    return (
        float(row.get("lpips") or "inf"),
        float(row.get("relative_l1") or "inf"),
        -float(row.get("ssim") or "0"),
        float(row.get("rmse") or "inf"),
    )


def best_keep_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    keep_rows = iter_keep_rows(rows)
    if not keep_rows:
        return None
    return min(keep_rows, key=metric_key)


def config_key_from_row(row: dict[str, str]) -> tuple[int, int, int, int, bool, str]:
    return (
        int(row["pade_m"]),
        int(row["pade_n"]),
        int(row["max_order"]),
        int(row["interval"]),
        (row.get("pade_only_single_step", "true").lower() != "false"),
        format(float(row.get("pade_denom_threshold") or "1e-4"), ".6g"),
    )


def load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_latest_run_summary() -> tuple[Path | None, dict[str, object] | None, dict[str, object] | None]:
    if not RUNS_DIR.exists():
        return None, None, None
    run_dirs = [path for path in RUNS_DIR.iterdir() if path.is_dir()]
    if not run_dirs:
        return None, None, None

    completed_dirs = [path for path in run_dirs if (path / "train_summary.json").exists()]
    if completed_dirs:
        latest_dir = max(completed_dirs, key=lambda path: path.name)
        return (
            latest_dir,
            load_json(latest_dir / "train_summary.json"),
            load_json(latest_dir / "approx_stats.json"),
        )

    latest_dir = max(run_dirs, key=lambda path: path.name)
    return latest_dir, None, None


def latest_ratio(
    summary: dict[str, object] | None,
    approx_stats: dict[str, object] | None,
    field: str,
) -> float | None:
    for payload in (approx_stats, summary):
        if not payload:
            continue
        value = payload.get(field)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def make_candidate(
    description: str,
    pade_m: int,
    pade_n: int,
    *,
    max_order: int | None = None,
    interval: int = 4,
    pade_only_single_step: bool = True,
    pade_denom_threshold: float = 1e-4,
) -> Candidate:
    return Candidate(
        description=description,
        pade_m=pade_m,
        pade_n=pade_n,
        max_order=max_order if max_order is not None else pade_m + pade_n,
        interval=interval,
        pade_only_single_step=pade_only_single_step,
        pade_denom_threshold=pade_denom_threshold,
    )


def base_catalog() -> list[Candidate]:
    return [
        make_candidate("auto baseline [1/1] pade 100img", 1, 1, max_order=4),
        make_candidate("auto candidate [2/1] pade 100img", 2, 1, max_order=3),
        make_candidate("auto candidate [1/2] pade 100img", 1, 2, max_order=3),
        make_candidate("auto candidate [2/2] pade 100img", 2, 2, max_order=4),
        make_candidate("auto candidate [3/1] pade 100img", 3, 1, max_order=4),
        make_candidate("auto candidate [1/3] pade 100img", 1, 3, max_order=4),
        make_candidate("auto candidate [3/2] pade 100img", 3, 2, max_order=5),
        make_candidate("auto candidate [2/3] pade 100img", 2, 3, max_order=5),
        make_candidate("auto candidate [4/1] pade 100img", 4, 1, max_order=5),
    ]


def threshold_sweep(best_row: dict[str, str]) -> list[Candidate]:
    pade_m = int(best_row["pade_m"])
    pade_n = int(best_row["pade_n"])
    max_order = int(best_row["max_order"])
    interval = int(best_row["interval"])
    thresholds = [3e-4, 1e-3, 3e-3, 1e-2]
    candidates = []
    for threshold in thresholds:
        candidates.append(
            make_candidate(
                f"auto threshold {threshold:.0e} [{pade_m}/{pade_n}] pade 100img",
                pade_m,
                pade_n,
                max_order=max_order,
                interval=interval,
                pade_only_single_step=True,
                pade_denom_threshold=threshold,
            )
        )
    return candidates


def single_step_relaxation(best_row: dict[str, str]) -> Candidate:
    pade_m = int(best_row["pade_m"])
    pade_n = int(best_row["pade_n"])
    max_order = int(best_row["max_order"])
    interval = int(best_row["interval"])
    threshold = float(best_row.get("pade_denom_threshold") or "1e-4")
    return make_candidate(
        f"auto relax single-step [{pade_m}/{pade_n}] pade 100img",
        pade_m,
        pade_n,
        max_order=max_order,
        interval=interval,
        pade_only_single_step=False,
        pade_denom_threshold=threshold,
    )


def local_neighbors(best_row: dict[str, str]) -> list[Candidate]:
    pade_m = int(best_row["pade_m"])
    pade_n = int(best_row["pade_n"])
    interval = int(best_row["interval"])
    threshold = float(best_row.get("pade_denom_threshold") or "1e-4")
    seen = set()
    candidates = []
    for m, n in [
        (pade_m + 1, pade_n),
        (pade_m, pade_n + 1),
        (pade_m + 1, pade_n + 1),
        (max(1, pade_m - 1), pade_n + 1),
        (pade_m + 1, max(1, pade_n - 1)),
    ]:
        if (m, n) in seen:
            continue
        seen.add((m, n))
        candidates.append(
            make_candidate(
                f"auto local [{m}/{n}] pade 100img",
                m,
                n,
                interval=interval,
                pade_only_single_step=True,
                pade_denom_threshold=threshold,
            )
        )
    return candidates


def choose_next_candidate(
    rows: list[dict[str, str]],
    best_row: dict[str, str] | None,
    latest_summary: dict[str, object] | None,
    latest_approx_stats: dict[str, object] | None,
) -> Candidate:
    tried = {config_key_from_row(row) for row in rows}

    for candidate in base_catalog():
        if candidate.key not in tried:
            return candidate

    if best_row is None:
        return make_candidate("auto baseline [1/1] pade 100img", 1, 1)

    pade_step_ratio = latest_ratio(latest_summary, latest_approx_stats, "pade_step_ratio")
    pade_call_ratio = latest_ratio(latest_summary, latest_approx_stats, "pade_call_ratio")
    ratios_low = (
        pade_step_ratio is not None
        and pade_call_ratio is not None
        and max(pade_step_ratio, pade_call_ratio) < 0.05
    )

    if ratios_low:
        for candidate in threshold_sweep(best_row):
            if candidate.key not in tried:
                return candidate

    relaxed = single_step_relaxation(best_row)
    if relaxed.key not in tried:
        return relaxed

    for candidate in local_neighbors(best_row):
        if candidate.key not in tried:
            return candidate

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return Candidate(
        description=f"auto revisit best [{best_row['pade_m']}/{best_row['pade_n']}] {timestamp}",
        pade_m=int(best_row["pade_m"]),
        pade_n=int(best_row["pade_n"]),
        max_order=int(best_row["max_order"]),
        interval=int(best_row["interval"]),
        pade_only_single_step=(best_row.get("pade_only_single_step", "true").lower() != "false"),
        pade_denom_threshold=float(best_row.get("pade_denom_threshold") or "1e-4"),
    )


def replace_field(text: str, field: str, value: str) -> str:
    pattern = re.compile(rf"^(\s*{field}:\s*[^=]+=\s*).*$", re.MULTILINE)
    updated, count = pattern.subn(lambda match: f"{match.group(1)}{value}", text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not update {field} in train.py")
    return updated


def patch_train_py(candidate: Candidate) -> None:
    text = TRAIN_PY.read_text(encoding="utf-8")
    replacements = {
        "description": json.dumps(candidate.description),
        "pade_m": str(candidate.pade_m),
        "pade_n": str(candidate.pade_n),
        "max_order": str(candidate.max_order),
        "interval": str(candidate.interval),
        "enable_pade": "True" if candidate.enable_pade else "False",
        "pade_only_single_step": "True" if candidate.pade_only_single_step else "False",
        "pade_denom_threshold": format(candidate.pade_denom_threshold, ".6g"),
    }
    updated = text
    for field, value in replacements.items():
        updated = replace_field(updated, field, value)
    if updated != text:
        TRAIN_PY.write_text(updated, encoding="utf-8")


def run_checked(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )


def train_py_has_uncommitted_changes() -> bool:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--quiet", "--", str(TRAIN_PY)],
        check=False,
    )
    return result.returncode == 1


def commit_train_config(candidate: Candidate) -> None:
    if not train_py_has_uncommitted_changes():
        print("[loop] train.py already at requested config; skipping commit", flush=True)
        return
    run_checked(["git", "-C", str(ROOT), "add", str(TRAIN_PY)])
    message = (
        f"auto pade m{candidate.pade_m} n{candidate.pade_n} "
        f"k{candidate.max_order} i{candidate.interval}"
    )
    run_checked(["git", "-C", str(ROOT), "commit", "-m", message, "--", str(TRAIN_PY)])


def parse_run_summary() -> dict[str, str]:
    if not RUN_LOG.exists():
        return {}
    summary: dict[str, str] = {}
    in_summary = False
    for raw_line in RUN_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if line == "---":
            summary = {}
            in_summary = True
            continue
        if not in_summary or ": " not in line:
            continue
        key, value = line.split(": ", 1)
        summary[key] = value
    return summary


def print_iteration_state(
    iteration: int,
    rows: list[dict[str, str]],
    best_row: dict[str, str] | None,
    latest_dir: Path | None,
    latest_summary: dict[str, object] | None,
    latest_approx_stats: dict[str, object] | None,
) -> None:
    print(f"[loop] iteration={iteration}", flush=True)
    print(f"[loop] results_rows={len(rows)}", flush=True)
    if best_row is None:
        print("[loop] best_keep=none", flush=True)
    else:
        print(
            "[loop] best_keep="
            f"m{best_row['pade_m']} n{best_row['pade_n']} "
            f"k{best_row['max_order']} i{best_row['interval']} "
            f"lpips={best_row.get('lpips', '')} "
            f"relative_l1={best_row.get('relative_l1', '')} "
            f"ssim={best_row.get('ssim', '')} "
            f"rmse={best_row.get('rmse', '')}",
            flush=True,
        )
    if latest_dir is None:
        print("[loop] latest_run=none", flush=True)
        return
    print(f"[loop] latest_run={latest_dir}", flush=True)
    if latest_summary is not None:
        status = latest_summary.get("status", "")
        lpips = latest_summary.get("lpips", "")
        step_ratio = latest_ratio(latest_summary, latest_approx_stats, "pade_step_ratio")
        call_ratio = latest_ratio(latest_summary, latest_approx_stats, "pade_call_ratio")
        print(
            f"[loop] latest_status={status} lpips={lpips} "
            f"pade_step_ratio={step_ratio if step_ratio is not None else 'n/a'} "
            f"pade_call_ratio={call_ratio if call_ratio is not None else 'n/a'}",
            flush=True,
        )


def main() -> int:
    iteration = 1
    while True:
        rows = read_results()
        best_row = best_keep_row(rows)
        latest_dir, latest_summary, latest_approx_stats = load_latest_run_summary()
        print_iteration_state(
            iteration,
            rows,
            best_row,
            latest_dir,
            latest_summary,
            latest_approx_stats,
        )

        candidate = choose_next_candidate(rows, best_row, latest_summary, latest_approx_stats)
        print(
            "[loop] next_candidate="
            f"m{candidate.pade_m} n{candidate.pade_n} "
            f"k{candidate.max_order} i{candidate.interval} "
            f"single_step={candidate.pade_only_single_step} "
            f"threshold={format(candidate.pade_denom_threshold, '.6g')}",
            flush=True,
        )

        patch_train_py(candidate)
        commit_train_config(candidate)

        with RUN_LOG.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                [str(PYTHON), str(TRAIN_PY)],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )

        summary = parse_run_summary()
        print(
            f"[loop] run_exit={completed.returncode} "
            f"status={summary.get('status', 'unknown')} "
            f"lpips={summary.get('lpips', 'n/a')} "
            f"relative_l1={summary.get('relative_l1', 'n/a')} "
            f"ssim={summary.get('ssim', 'n/a')} "
            f"rmse={summary.get('rmse', 'n/a')}",
            flush=True,
        )
        print(
            f"[loop] pade_step_ratio={summary.get('pade_step_ratio', 'n/a')} "
            f"pade_call_ratio={summary.get('pade_call_ratio', 'n/a')} "
            f"artifact_dir={summary.get('artifact_dir', '')}",
            flush=True,
        )

        if completed.returncode != 0:
            print("[loop] run crashed; continuing to next iteration", flush=True)

        iteration += 1


if __name__ == "__main__":
    raise SystemExit(main())
