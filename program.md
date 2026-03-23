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

Improve TaylorSeer-DiT sampling quality by iteratively editing the Padé implementation itself.
Do not treat Padé runtime parameters as the active search space for now.

The runtime configuration is fixed to the following baseline taken from run `20260319-234247`:

- `pade_m=1`
- `pade_n=2`
- `max_order=3`
- `interval=4`
- `pade_only_single_step=True`
- `pade_denom_threshold=1e-3`
- `total_images=100`

The intended operating mode is an indefinitely running agent-driven search loop that continues until manually interrupted.
The agent must remain the decision-maker for candidate selection throughout the search.

Primary search surface:

- gating logic in `/home/yjs/TaylorSeer/TaylorSeer-DiT/taylor_utils/__init__.py`
- Padé coefficient construction
- denominator stabilization
- fallback and blending behavior
- order-selection logic
- numerical guards

Prefer small, defensible code changes. Do not randomly thrash across unrelated ideas.

## Non-Negotiable Constraints

These rules are strict. Follow them unless the human explicitly overrides them.

1. Keep the search loop agent-driven.
   - Do not delegate search control to `pade_search_loop.py` or any other loop-controller, supervisor, planner, driver, or search-runner script.
   - Do not create a durable local controller that chooses candidates without the agent.
   - The agent must inspect disk state, choose the next candidate, and decide each iteration itself.
2. Keep `train.py` as the single-run experiment entrypoint unless there is a clear technical reason to change it.
   - Avoid redesigning the framework.
   - Use `train.py` to execute one run under the fixed runtime configuration.
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
8. Keep the runtime parameters fixed.
   - Do not search over `pade_m`, `pade_n`, `max_order`, `interval`, `pade_only_single_step`, or `pade_denom_threshold`.
   - Only the human may explicitly re-open parameter search.

## Metric priority

Use this ranking when deciding whether a run is better:

1. lower `LPIPS`
2. if tied, lower `Relative L1`
3. if tied, higher `SSIM`
4. if tied, lower `RMSE`

`train.py` already uses this ordering when marking a successful run as `keep` or `discard`.

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

Typical code regions to improve:

- `_should_use_pade`
- `pade_formula_11`
- `pade_coefficients_from_taylor`
- `choose_pade_order_by_available_derivatives`
- `_evaluate_generic_pade`
- stability guards and blending rules

Use `train.py` as the fixed single-run harness.
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

The baseline is the fixed runtime config already encoded in `train.py`, corresponding to the settings used in run `20260319-234247`.

Launch one iteration with:

```bash
/home/yjs/xdit_env/bin/python train.py   --description "baseline fixed-parameter Padé code"   > run.log 2>&1
```

Read the summary with:

```bash
grep '^status:\|^lpips:\|^relative_l1:\|^ssim:\|^rmse:' run.log
```

## Loop

LOOP FOREVER until the human stops you:

1. Re-read `results.tsv` from disk at the start of every loop iteration, and also inspect the latest directory under `runs/` so you recover state from files instead of relying on chat memory.
2. Keep context usage low:
   - do not repeatedly read entire large logs or the full `results.tsv` into context unless there is a specific need
   - prefer compact summaries extracted from files
   - for `results.tsv`, focus on the current best kept rows, the most recent rows, and any clear code-change trends
   - for `run.log`, read the final summary first, and only read traceback lines if the run crashed
   - for detailed metrics and code provenance, prefer the structured files in the latest run directory such as `train_summary.json`, `eval_metrics.json`, `approx_stats.json`, and `pade_target_snapshot.py`
3. Identify the current best kept implementation.
4. Choose exactly one next code candidate for the next experiment.
   - Do not batch-edit multiple independent ideas in advance.
   - Do not hand this decision to a script.
5. Update the Padé implementation in `/home/yjs/TaylorSeer/TaylorSeer-DiT/taylor_utils/__init__.py`.
   - Prefer one coherent hypothesis per iteration.
   - Keep diffs small enough that failures are attributable.
6. Update `description` in the next run if that helps traceability.
7. Commit the change when it is useful for traceability.
8. Run one experiment:

```bash
/home/yjs/xdit_env/bin/python train.py   --description "tighten denominator guard and blend toward Taylor near poles"   > run.log 2>&1
```

9. Read the summary from `run.log`.
10. If the run crashed, inspect the traceback in `run.log`.
11. Read the newest row in `results.tsv`.
12. Read the newest `train_summary.json`.
13. Read the newest `approx_stats.json`.
14. Read `pade_code_sha256` and `pade_snapshot_file` from the latest summary when you need to confirm exactly which Padé implementation was evaluated.
15. Use the result to decide the next code candidate.
16. Continue immediately within the active agent workflow.

Do not pause to ask whether to continue.
Do not pause to ask whether to apply the next candidate's code changes.
Do not convert the workflow into a local controller just to survive chat-turn boundaries.

## Search strategy

Use disciplined local code search.

Suggested order:

1. establish a stable baseline for the current Padé implementation
2. improve one numerical issue at a time
3. prefer conservative stability fixes before aggressive higher-order logic
4. keep the number of moving parts per candidate small
5. favor changes that improve both quality and robustness over brittle one-off wins
6. use `pade_step_ratio` and `pade_call_ratio` to verify whether Padé is actually activating
7. if Padé rarely activates, improve the gate or eligibility logic before drawing strong conclusions about the approximation formula
8. if Padé activates often but quality worsens, focus on coefficient construction, denominator safety, or blending back toward Taylor
9. if a candidate crashes due to an obvious implementation issue, fix the implementation and continue; do not abandon the overall loop

## Output

Each run prints a stable summary that starts with `---`.

It includes:

- `status`
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
- `pade_only_single_step`
- `pade_denom_threshold`
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
- `keep`, `discard`, or `crash`
- key metrics: `lpips`, `relative_l1`, `ssim`, `rmse`
- `pade_step_ratio`
- `pade_call_ratio`
- the next planned code candidate

Immediately after reporting, start the next run unless an allowed blocker from the execution model is present.

## Notes

- `prepare.py` from the original autoresearch project is not part of this task.
- The original language-model training workflow is not used here.
- For this repo, `train.py` serves as the single experiment runner, and the agent remains responsible for orchestration.
