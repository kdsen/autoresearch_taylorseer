# autoresearch for TaylorSeer Padé code optimization

This repo is being used as an autonomous research harness for TaylorSeer-DiT.

The experiment entrypoint is `train.py` in this repo, but it does not train a language model anymore. It runs one TaylorSeer Padé experiment end to end:

1. clears old generated samples
2. runs `/home/yjs/TaylorSeer/TaylorSeer-DiT/sample.py`
3. runs `/home/yjs/eval_image_diff.py`
4. reads the metrics
5. appends one row to `results.tsv`
6. writes run artifacts under `runs/`

## In-scope files

Read these files first:

- `README.md`
- `program.md`
- `train.py`

Primary code under optimization:

- `/home/yjs/TaylorSeer/TaylorSeer-DiT/taylor_utils/__init__.py`

You may inspect these external files as needed:

- `/home/yjs/TaylorSeer/TaylorSeer-DiT/sample.py`
- `/home/yjs/TaylorSeer/TaylorSeer-DiT/sample_ddp.py`
- `/home/yjs/eval_image_diff.py`

## Objective

Improve TaylorSeer-DiT sampling quality by iteratively improving the Padé implementation under a fixed evaluation bucket.
Do not keep mixing Padé runtime-parameter search with Padé code search during the current phase. First make Padé beat a standing pure Taylor baseline at a fixed `interval=3` bucket, then consider reopening a small parameter search later.

Use the following fixed evaluation bucket for the current phase:

- choose exactly one Padé order per run: either `pade_m=1, pade_n=2` or `pade_m=2, pade_n=1`
- `max_order=3`
- `interval=3`
- `pade_only_single_step=True`
- `pade_denom_threshold=1e-3`
- `total_images=100`

Use run `20260329-171615` as the standing pure Taylor baseline reference for this phase.
That baseline is valid because it matches the Taylor-affecting runtime settings for the fixed `interval=3`, `max_order=3`, `total_images=100` bucket.
Do not rerun pure Taylor every iteration unless one of the following is true:

- the code change touches shared Taylor-path logic rather than Padé-only logic
- a Padé candidate clearly beats the standing baseline and needs confirmation against a same-code pure Taylor control

The intended operating mode is an indefinitely running agent-driven search loop that continues until manually interrupted.
The agent must remain the decision-maker for candidate selection throughout the search.

Primary search surface:

- gating logic in `/home/yjs/TaylorSeer/TaylorSeer-DiT/taylor_utils/__init__.py`
- Padé coefficient construction
- denominator stabilization
- fallback and blending behavior
- order-selection logic
- numerical guards
- result attribution and standing-baseline comparison logic in `train.py`

Prefer small, defensible code changes. Do not randomly thrash across unrelated ideas.

## Non-Negotiable Constraints

These rules are strict. Follow them unless the human explicitly overrides them.

1. Keep the search loop agent-driven.
   - Do not delegate search control to `pade_search_loop.py` or any other loop-controller, supervisor, planner, driver, or search-runner script.
   - Do not create a durable local controller that chooses candidates without the agent.
   - The agent must inspect disk state, choose the next candidate, and decide each iteration itself.
2. Keep `train.py` as the single-run experiment entrypoint unless there is a clear technical reason to change it.
   - Avoid redesigning the framework.
   - Use `train.py` to execute one run under the currently chosen Padé code and runtime configuration.
   - During the current phase, do not keep editing the Padé-related `ExperimentConfig` defaults just to search parameters.
   - It is allowed to edit `train.py` when the attribution logic, standing-baseline comparison, or result bookkeeping needs improvement.
   - Do not move the search policy into code.
3. Recover state from disk, not from chat memory.
   - At the start of every iteration, re-read `results.tsv`.
   - Inspect the latest run directory under `runs/`.
   - Prefer `train_summary.json`, `eval_metrics.json`, `approx_stats.json`, and `pade_target_snapshot.py` over long logs.
4. Keep running until manually interrupted.
   - Do not ask whether to continue.
   - Do not ask whether to apply the next Padé code change.
   - Treat Padé implementation edits and description-only metadata changes as pre-authorized by this program.
   - Do not stop just because one local code idea fails.
   - If a run crashes, diagnose it, log it through the normal flow, and continue.
   - A per-run report is only a heartbeat, not a handoff point.
   - Do not return a final answer just because one run completed.
5. Prefer foreground execution.
   - Prefer a foreground PTY session that the agent can actively monitor and continue from in the same turn.
   - Do not intentionally daemonize, detach, `nohup`, or otherwise hand off the loop to a background controller.
   - If the environment stops the agent despite these instructions, recover from disk on the next start rather than replacing the workflow with a local controller.
6. Do not delete existing runs or results.
7. Do not revert unrelated local changes.
8. Freeze Padé runtime parameters during the current code-search phase.
   - Keep `max_order=3`, `interval=3`, `pade_only_single_step=True`, `pade_denom_threshold=1e-3`, and `total_images=100` fixed.
   - For Padé order, allow exactly two choices in this phase: `[1/2]` and `[2/1]`.
   - Treat `[1/2]` and `[2/1]` as separate comparable buckets; do not compare them as if they were the same configuration.
   - Keep `seed`, `cfg_scale`, `num_sampling_steps`, and `batch_size` fixed as encoded in `train.py`.
   - Do not reopen parameter search unless the human explicitly asks, or Padé has already produced stable wins over the standing pure Taylor baseline.
   - Do not combine a runtime-parameter change with an unrelated Padé implementation change in the same evaluation.

## Metric priority

Use this ranking when deciding whether a run is better:

1. lower `LPIPS`
2. if tied, lower `Relative L1`
3. if tied, higher `SSIM`
4. if tied, lower `RMSE`

`train.py` should use this ordering only inside comparable buckets.
For Padé candidates in the current phase, the primary question is whether they beat the standing pure Taylor baseline under the fixed interval-3 bucket.

## Setup

Before starting the loop:

1. Create a fresh branch such as `autoresearch/pade-code-<date>`.
2. Verify these paths exist:
   - `/home/yjs/xdit_env/bin/python`
   - `/home/yjs/TaylorSeer/TaylorSeer-DiT/sample.py`
   - `/home/yjs/TaylorSeer/TaylorSeer-DiT/taylor_utils/__init__.py`
   - `/home/yjs/eval_image_diff.py`
   - `/home/yjs/baseline_samples`
3. Verify imports work in `/home/yjs/xdit_env`.
4. Confirm `results.tsv` exists or let `train.py` create it.
5. Confirm `runs/` exists or let `train.py` create it.

## What to edit

Normal candidate iterations should primarily edit `/home/yjs/TaylorSeer/TaylorSeer-DiT/taylor_utils/__init__.py`.
Only edit `train.py` when the attribution harness itself needs improvement.

Typical code regions to improve:

- `_should_use_pade`
- `pade_formula_11`
- `pade_coefficients_from_taylor`
- `choose_pade_order_by_available_derivatives`
- `_evaluate_generic_pade`
- stability guards and blending rules

Use `train.py` as the fixed single-run harness for a standing-baseline Padé code search.
Do not redesign the harness unless the human explicitly asks for framework changes.
Do not use `pade_search_loop.py` or add any new loop-controller or supervisor.

## Execution model

The command below executes exactly one experiment iteration:

```bash
/home/yjs/xdit_env/bin/python train.py > run.log 2>&1
```

That command is intentionally single-run. The indefinite loop is created by the agent reissuing this one-shot command after each completed post-run review.

Required behavior:

- Keep the loop alive by repeatedly choosing one code candidate, running the single-run command once, reading the resulting artifacts, and immediately choosing the next candidate.
- Treat the active agent session as the loop owner. Prefer keeping the workflow alive in a foreground PTY or equivalent actively monitored session.
- Do not replace this with a shell `while` loop, `nohup`, daemon, background service, cron job, local controller, or search-runner script.
- Do not reinterpret the single-run command itself as the loop. The loop is the agent's repeated decision-run-review cycle.
- If the environment interrupts the session or ends the turn, resume by rereading `results.tsv` and the latest run directory from disk, then continue the loop immediately.
- Do not ask for confirmation before launching the next candidate if the only changes are Padé code edits or description text.
- Do not wait for user approval after printing a run summary.
- After each summary, immediately decide the next candidate and launch it in the same turn unless there is an environment blocker.
- The only valid reasons to stop and ask the human are:
  1. missing required paths
  2. import or interpreter failure that blocks all runs
  3. CUDA or GPU unavailable
  4. repeated identical crash with no clear local fix
  5. an explicit human interruption

## Baseline

The current baseline protocol is:

- fixed interval-3 evaluation bucket from `train.py`
- allowed Padé orders for the current phase: `[1/2]` and `[2/1]`
- standing pure Taylor reference from run `20260329-171615`
- no repeated pure Taylor rerun for every Padé hypothesis
- rerun pure Taylor only for confirmation when needed

Launch one Padé candidate iteration with:

```bash
/home/yjs/xdit_env/bin/python train.py   --enable-pade   --pade-m 1   --pade-n 2   --description "candidate: stricter Padé abstention gate on frozen interval-3 bucket [1/2]"   > run.log 2>&1
```

or

```bash
/home/yjs/xdit_env/bin/python train.py   --enable-pade   --pade-m 2   --pade-n 1   --description "candidate: stricter Padé abstention gate on frozen interval-3 bucket [2/1]"   > run.log 2>&1
```

Read the summary with:

```bash
grep '^status:\|^paired_outcome:\|^paired_control_timestamp:\|^lpips:\|^relative_l1:\|^ssim:\|^rmse:' run.log
```

If a Padé candidate needs confirmation against a same-code pure Taylor control, launch:

```bash
/home/yjs/xdit_env/bin/python train.py   --no-enable-pade   --description "confirm control: hypothesis X on frozen interval-3 bucket"   > run.log 2>&1
```

## Loop

LOOP FOREVER until the human stops you:

1. Re-read `results.tsv` from disk at the start of every loop iteration, and also inspect the latest directory under `runs/` so you recover state from files instead of relying on chat memory.
2. Keep context usage low:
   - do not repeatedly read entire large logs or the full `results.tsv` into context unless there is a specific need
   - prefer compact summaries extracted from files
   - for `results.tsv`, focus on the standing pure Taylor reference, the best Padé rows inside the fixed bucket, the most recent rows, and any clear code-change trends
   - for `run.log`, read the final summary first, and only read traceback lines if the run crashed
   - for detailed metrics and code provenance, prefer the structured files in the latest run directory such as `train_summary.json`, `eval_metrics.json`, `approx_stats.json`, and `pade_target_snapshot.py`
3. Identify the current standing pure Taylor baseline and the current best Padé run for each allowed Padé order bucket.
4. Choose exactly one next Padé code candidate for the next experiment.
   - Do not batch-edit multiple independent ideas in advance.
   - Do not hand this decision to a script.
5. Update `/home/yjs/TaylorSeer/TaylorSeer-DiT/taylor_utils/__init__.py` with one coherent implementation hypothesis.
   - Prefer one coherent hypothesis per iteration.
   - Keep diffs small enough that failures are attributable.
6. Prefer stronger abstention gates, stricter safety checks, and better observability over broader Padé coverage.
7. Update `description` in the next run if that helps traceability.
8. Commit the change when it is useful for traceability.
9. Run one Padé experiment:

```bash
/home/yjs/xdit_env/bin/python train.py   --enable-pade   --pade-m 1   --pade-n 2   --description "tighten Padé abstention gate on frozen interval-3 bucket [1/2]"   > run.log 2>&1
```

10. Read the summary from `run.log`.
11. If the run crashed, inspect the traceback in `run.log`.
12. Read the newest row in `results.tsv`.
13. Read the newest `train_summary.json`.
14. Read the newest `approx_stats.json`.
15. Read `pade_code_sha256`, `pade_snapshot_file`, `paired_control_scope`, `paired_control_timestamp`, and `paired_outcome` from the latest summary.
16. If the code change touched shared Taylor-path logic, or the candidate clearly beats the standing baseline, run one confirmation control with `--no-enable-pade` before treating the gain as real.
17. Use the result to decide the next Padé code candidate.
18. Continue immediately within the active agent workflow.

Do not pause to ask whether to continue.
Do not pause to ask whether to apply the next candidate's code changes.
Do not convert the workflow into a local controller just to survive chat-turn boundaries.

## Search strategy

Use disciplined local code search.

Suggested order:

1. treat `20260329-171615` as the standing pure Taylor reference for the fixed interval-3 bucket
2. treat `[1/2]` and `[2/1]` as the only two allowed Padé order options in this phase
3. compare Padé candidates within their own order bucket instead of mixing `[1/2]` and `[2/1]` runs together
4. change only one implementation hypothesis per iteration
5. prefer stronger Padé abstention and safety gates before trying more aggressive rational approximations
6. keep the number of moving parts per candidate small
7. favor changes that improve quality relative to the standing baseline, not changes that merely increase Padé activation rate
8. use `pade_step_ratio` and `pade_call_ratio` to verify whether Padé is activating, but do not optimize those ratios directly
9. only reopen runtime-parameter search after Padé code is consistently competitive with or better than the standing pure Taylor reference
10. if Padé activates often but quality worsens, focus on coefficient construction, denominator safety, blending back toward Taylor, or a nearby order/threshold adjustment
11. if a candidate crashes due to an obvious implementation issue, fix the implementation and continue; do not abandon the overall loop

## Output

Each run prints a stable summary that starts with `---`.

It includes:

- `status`
- `comparison_bucket`
- `bucket_best`
- `pade_target_file`
- `pade_code_sha256`
- `pade_snapshot_file`
- `lpips`
- `relative_l1`
- `ssim`
- `rmse`
- `psnr`
- `cosine`
- `sample_seconds`
- `eval_seconds`
- `total_seconds`
- `pade_m`
- `pade_n`
- `max_order`
- `interval`
- `enable_pade`
- `pade_only_single_step`
- `pade_denom_threshold`
- `total_images`
- `batch_size`
- `seed`
- `cfg_scale`
- `num_sampling_steps`
- `paired_control_scope`
- `paired_control_timestamp`
- `paired_outcome`
- `paired_control_lpips`
- `paired_control_relative_l1`
- `paired_control_ssim`
- `paired_control_rmse`
- `paired_delta_lpips`
- `paired_delta_relative_l1`
- `paired_delta_ssim`
- `paired_delta_rmse`
- `approx_total_steps`
- `pade_steps`
- `taylor_steps`
- `mixed_steps`
- `pade_calls`
- `taylor_calls`
- `pade_step_ratio`
- `pade_call_ratio`
- `artifact_dir`

`train.py` also appends one row to `results.tsv` automatically.

## Reporting

Keep progress reports concise. For each completed run, include:

- the code candidate just tested
- the short `pade_code_sha256`
- the `status`, especially whether the Padé run beats, ties, or loses to the standing pure Taylor reference
- key metrics: `lpips`, `relative_l1`, `ssim`, `rmse`
- `paired_outcome`
- `pade_step_ratio`
- `pade_call_ratio`
- the next planned code candidate

Immediately after reporting, start the next run unless an allowed blocker from the execution model is present.

## Notes

- `prepare.py` from the original autoresearch project is not part of this task.
- The original language-model training workflow is not used here.
- For this repo, `train.py` serves as the single experiment runner, and the agent remains responsible for orchestration.
