# autoresearch for TaylorSeer approximation-family optimization

This repo is being used as an autonomous research harness for TaylorSeer-DiT.

The experiment entrypoint is `train.py` in this repo. It runs one approximation-family experiment end to end:

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

Stop treating the current search as a Padé-optimization task.
Use `pade_formula_mn()` as the implementation hook for a broader approximation-family search that can represent a better alternative to the current TaylorSeer approximation path.
Allowed approximation families include, but are not limited to:

- low-order polynomials
- rational functions
- piecewise functions
- hybrid or gated mixtures
- conservative blends that fall back to Taylor when unsafe

The optimization target for this phase is:

1. stay within the standing latency budget derived from the pure TaylorSeer baseline
2. beat that baseline on image-quality metrics

Use run `20260329-171615` as the standing baseline reference for this phase.
That baseline has:

- `lpips=0.061478`
- `relative_l1=1.056978`
- `ssim=0.899468`
- `rmse=5.597430`
- `sample_seconds=751.529`

Treat "sample_seconds at or around baseline" as a hard latency budget of at most `1.05x` the standing baseline, i.e. at most about `789.105` seconds.
A candidate that improves quality but clearly exceeds that latency budget does not count as a success for this phase.

## Fixed evaluation bucket

Freeze the runtime bucket during this phase:

- `pade_m=1`
- `pade_n=2`
- `max_order=3`
- `interval=3`
- `pade_only_single_step=True`
- `pade_denom_threshold=1e-3`
- `total_images=100`
- `batch_size=2`
- `seed=42`
- `cfg_scale=1.5`
- `num_sampling_steps=250`

Do not reopen runtime-parameter search during this phase.
Do not combine a runtime-parameter change with an approximation-family code change in the same evaluation.

## Primary search surface

Normal candidate iterations must keep the implementation scope as narrow as possible.
During this phase, the intended editable search surface is:

- `/home/yjs/TaylorSeer/TaylorSeer-DiT/taylor_utils/__init__.py`
- specifically the function `pade_formula_mn()`

Treat `pade_formula_mn()` as the single approximation hook to iterate on.
The objective is not to preserve Padé semantics; the objective is to discover a better function family behind that hook.

Do not spread the search across unrelated functions unless a change to `pade_formula_mn()` strictly requires a tiny helper adjustment in the same file.
Do not redesign the broader framework.

## Non-Negotiable Constraints

These rules are strict unless the human explicitly overrides them.

1. Keep the search loop agent-driven.
   - Do not delegate search control to `pade_search_loop.py` or any other controller script.
   - The agent must inspect disk state, choose the next candidate, and decide each iteration itself.
2. Keep `train.py` as the single-run experiment entrypoint.
   - It may be edited only for result bookkeeping, baseline comparison, or latency-budget logic.
   - Do not move the search policy into code.
3. Recover state from disk, not from chat memory.
   - Re-read `results.tsv` and inspect the latest `runs/` directory.
4. Keep running until manually interrupted.
   - Do not ask whether to continue.
   - Do not ask whether to apply the next `pade_formula_mn()` change.
5. Prefer foreground execution.
   - Do not intentionally daemonize or hand off the loop to a background controller.
6. Do not delete existing runs or results.
7. Do not revert unrelated local changes.
8. Keep the implementation scope narrow.
   - For ordinary candidate iterations, edit only `taylor_utils/__init__.py`.
   - Within that file, prefer to edit only `pade_formula_mn()`.

## Metric policy

The decision order for candidate runs is:

1. first satisfy the latency budget relative to the standing baseline
2. among runs that satisfy the latency budget, rank by:
   - lower `LPIPS`
   - if tied, lower `Relative L1`
   - if tied, higher `SSIM`
   - if tied, lower `RMSE`

For this phase, a candidate is only a true win if it both:

- stays within the standing latency budget
- beats the standing baseline on the quality metrics above

## Setup

Before starting the loop:

1. Create or switch to a dedicated branch such as `autoresearch/function-family-<date>`.
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

Normal candidate iterations should edit:

- `/home/yjs/TaylorSeer/TaylorSeer-DiT/taylor_utils/__init__.py`
- specifically `pade_formula_mn()`

Only edit `train.py` when the attribution harness itself needs improvement.
Do not use `pade_search_loop.py` or add any new loop-controller or supervisor.

## Execution model

The command below executes exactly one experiment iteration:

```bash
/home/yjs/xdit_env/bin/python train.py > run.log 2>&1
```

That command is intentionally single-run. The indefinite loop is created by the agent reissuing this one-shot command after each completed post-run review.

Required behavior:

- Keep the loop alive by repeatedly choosing one approximation-family candidate, running the single-run command once, reading the resulting artifacts, and immediately choosing the next candidate.
- Treat the active agent session as the loop owner.
- Do not replace this with a shell `while` loop, daemon, background service, cron job, or local controller.
- If the environment interrupts the session, resume by rereading `results.tsv` and the latest run directory from disk.
- Do not ask for confirmation before launching the next candidate if the only changes are `pade_formula_mn()` edits or description text.

## Baseline and launch examples

The standing baseline is run `20260329-171615`.
Use it for both quality comparison and latency comparison.

Launch one candidate iteration with:

```bash
/home/yjs/xdit_env/bin/python train.py   --enable-pade   --pade-m 1   --pade-n 2   --description "candidate: piecewise rational-polynomial approximation under interval-3 latency budget"   > run.log 2>&1
```

Read the summary with:

```bash
grep '^status:\|^paired_outcome:\|^latency_within_budget:\|^sample_seconds:\|^sample_seconds_budget:\|^sample_seconds_ratio:\|^lpips:\|^relative_l1:\|^ssim:\|^rmse:' run.log
```

## Loop

LOOP FOREVER until the human stops you:

1. Re-read `results.tsv` and inspect the latest directory under `runs/`.
2. Keep context usage low. Prefer compact summaries from `results.tsv`, `train_summary.json`, `eval_metrics.json`, and `approx_stats.json`.
3. Treat `20260329-171615` as the standing baseline.
4. Identify the current best candidate that satisfies the latency budget.
5. Choose exactly one next approximation-family hypothesis for `pade_formula_mn()`.
6. Keep diffs small enough that failures are attributable.
7. Commit the change when useful for traceability.
8. Run one experiment.
9. Read the summary from `run.log`.
10. If the run crashed, inspect the traceback, log it, fix the obvious implementation issue, and continue.
11. Use the result to decide the next `pade_formula_mn()` candidate.
12. Continue immediately.

## Search strategy

Suggested order:

1. start with conservative families that are numerically safe and cheap
2. prefer piecewise or blended designs that fall back to Taylor when confidence is low
3. favor hypotheses that plausibly reduce error without increasing activation cost too much
4. keep only one implementation idea per iteration
5. if a candidate beats the baseline in quality but violates the latency budget, simplify it before exploring more aggressive families
6. use `pade_step_ratio` and `pade_call_ratio` only as diagnostics, not as primary optimization targets
7. if a candidate crashes due to an obvious implementation bug, fix the bug and continue the same search direction

## Candidate idea shortlist for `pade_formula_mn()`

To avoid search drift, bias candidate selection toward the following families first.
These are not mandatory, but they are the preferred search frontier for this phase.

1. Weighted Taylor family
   - Keep Taylor as the base approximation.
   - Apply conservative decay or reweighting to higher-order terms.
   - Prefer cheap, numerically stable variants first.

2. Taylor plus light rational correction
   - Use Taylor as the main path.
   - Add a small rational correction only when the local coefficients look safe.
   - Keep denominator handling conservative and cheap.

3. Piecewise approximation family
   - Use one formula in the safe region and fall back to Taylor in the risky region.
   - Safe/risky decisions may depend on step distance, coefficient magnitude, derivative ratios, denominator margin, or finite-value checks.

4. Blended hybrid family
   - Blend between Taylor and a stronger approximation using a bounded confidence weight.
   - Prefer smooth blending over hard switching when cost is similar.
   - The blend must collapse back toward Taylor when confidence is low.

5. Single-step-specialized family
   - Since the current runtime bucket is conservative and often emphasizes short extrapolation, it is acceptable to design approximations specialized for the single-step case first.
   - Do not optimize for broad generality before proving a win in the actual bucket.

## Anti-drift guidance for candidate selection

When choosing the next idea, prefer the following order:

1. smallest change to `pade_formula_mn()` that tests one approximation-family hypothesis
2. numerically safer candidate before more aggressive candidate
3. cheaper candidate before more expensive candidate when quality upside is unclear
4. fallback-to-Taylor design before fully replacing Taylor everywhere
5. latency-budget-compliant candidate before marginally better but slower candidate

Avoid drifting into the following unless the human explicitly asks:

- reopening runtime-parameter search
- redesigning `train.py` beyond bookkeeping or baseline logic
- broad edits across multiple functions in `__init__.py` when `pade_formula_mn()` alone can express the hypothesis
- optimizing Padé identity or theory for its own sake instead of optimizing the practical approximation family
- increasing complexity without a clear path to staying within the latency budget

## Output

Each run prints a stable summary that starts with `---`.
It includes the normal quality metrics and latency-aware comparison fields, including:

- `status`
- `paired_control_timestamp`
- `paired_outcome`
- `sample_seconds`
- `latency_baseline_timestamp`
- `latency_baseline_sample_seconds`
- `sample_seconds_budget`
- `sample_seconds_ratio`
- `latency_within_budget`
- `lpips`
- `relative_l1`
- `ssim`
- `rmse`
- `artifact_dir`

`train.py` also appends one row to `results.tsv` automatically.

## Reporting

Keep progress reports concise. For each completed run, include:

- the approximation-family candidate just tested
- the short `pade_code_sha256`
- whether it stayed within the latency budget
- whether it beat, tied, or lost to the standing baseline
- key metrics: `lpips`, `relative_l1`, `ssim`, `rmse`
- `sample_seconds` and `sample_seconds_ratio`
- the next planned candidate

Immediately after reporting, start the next run unless an allowed blocker is present.
