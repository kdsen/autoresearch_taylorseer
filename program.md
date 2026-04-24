# autoresearch for TaylorSeer approximation-family optimization

This repo is being used as an autonomous research harness for TaylorSeer-DiT.

中文速览：

- `program.md` 负责定义研究任务边界、评估口径和搜索纪律。
- `train.py` 是单次实验入口；它不负责长期调度，只负责把“一次候选实验”完整跑完并落盘。
- 真正被搜索和替换的近似实现，默认集中在 `/home/yjs/TaylorSeer/TaylorSeer-DiT/taylor_utils/__init__.py` 的 `pade_formula_mn()`。

The experiment entrypoint is `train.py` in this repo. It runs one approximation-family experiment end to end:

1. clears old generated samples
2. runs `/home/yjs/TaylorSeer/TaylorSeer-DiT/sample.py`
3. runs `/home/yjs/eval_image_diff.py`
4. reads the metrics
5. appends one row to `results.tsv`
6. writes run artifacts under `runs/`

对应中文流程：

1. 清理上一次采样残留，避免旧结果污染本次评估。
2. 调用 TaylorSeer 的 `sample.py` 生成当前候选近似函数的样本。
3. 调用 `eval_image_diff.py`，把新样本和基线样本做指标对比。
4. 读取评估指标与近似统计信息。
5. 结合延迟预算和基线结果，给本次运行打状态标签。
6. 把摘要写入 `results.tsv`，并把快照和评估产物归档到 `runs/`。

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

中文解释：

- 当前目标是找到一个能够在冻结 bucket 下真正替代 TaylorSeer 的新近似方法，而不是靠把近似缩到极少数 step、layer 或 module 上来“偷”一点点增益。
- 当前阶段仍然把 `pade_formula_mn()` 当成统一实验接口，但候选方法应在公式形态上与 TaylorSeer 有实质差异，并在非平凡作用范围内体现效果。
- `factors` 和 `x` 仍然是输入特征，但不要把门控、超稀疏启用、最后一步或倒数第二步特判当成主要创新来源。
- 如果一个候选方法的主要收益来自“更少使用近似”而不是“更好的近似公式”，它不应被视为主线成功。

Stop treating the current search as either a Padé-optimization task or a Taylor-term-retention task.
Use `pade_formula_mn()` as the implementation hook for discovering a genuinely stronger approximation family for this frozen bucket.
The input features may still come from the existing `factors` cache and step offset `x`, but the candidate formula should aim to replace TaylorSeer behavior on a meaningful subset of the bucket, not merely gate itself down to near-zero usage.
Allowed approximation families include, but are not limited to:

- ordinary low-order polynomials in the monomial basis
- Chebyshev polynomials
- Legendre polynomials
- Hermite-style cubic polynomials
- low-order rational-polynomial hybrids
- piecewise polynomials with a small number of interpretable regimes
- coefficient-construction rules from `factors` that materially change the approximation behavior

Gates are allowed only as secondary safety wrappers around a substantive approximation family.
A gate-only, sparsity-only, or scope-only trick does not count as a new family discovery.

The optimization target for this phase is:

1. discover a substantively different approximation family that can beat the pure TaylorSeer baseline on image-quality metrics
2. keep that family within the standing latency budget derived from the pure TaylorSeer baseline
3. among within-budget candidates, prefer larger and more robust quality gains over ultra-conservative near-tie wins
4. only treat a result as strategic progress if the gain comes from the formula itself rather than from sharply reducing where it is allowed to run

Use run `20260329-171615` as the standing baseline reference for this phase.
That baseline has:

- `lpips=0.061478`
- `relative_l1=1.056978`
- `ssim=0.899468`
- `rmse=5.597430`
- `sample_seconds=751.529`

Treat "sample_seconds at or around baseline" as a hard latency budget of at most `1.05x` the standing baseline, i.e. at most about `789.105` seconds.
A candidate that improves quality but clearly exceeds that latency budget does not count as a success for this phase.

Champion rule for this phase:

- Standing baseline determines whether a candidate is valid at all.
- A run is only a new search anchor if it is within budget, beats the standing baseline, and also beats the current best valid family candidate in the same frozen bucket.
- A near-tie that wins only by tiny metric noise should not end the search for a materially better family.
- A run that beats the standing baseline but does not beat the current family champion is informative, but it should not become the new retained champion or the main search anchor.

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

中文解释：

- 日常迭代时，主要只改一个地方：`pade_formula_mn()`。
- 这样每次实验的归因最清晰，便于从 `results.tsv` 和 `runs/` 回看“哪一个多项式族假设导致了什么结果”。

Normal candidate iterations must keep the implementation scope as narrow as possible.
During this phase, the intended editable search surface is:

- `/home/yjs/TaylorSeer/TaylorSeer-DiT/taylor_utils/__init__.py`
- specifically the function `pade_formula_mn()`

Treat `pade_formula_mn()` as the single approximation hook to iterate on.
The objective is not to preserve Padé semantics or Taylor retention semantics; the objective is to discover a better empirical polynomial family behind that hook.

Within that hook, it is acceptable to add a tiny helper in the same file when needed to express a polynomial basis cleanly, for example:

- basis normalization for `x`
- cheap basis evaluation for Chebyshev / Legendre / Hermite variants
- a compact coefficient-construction helper from `factors`

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
9. Do not silently drift back into Taylor-term damping or gate-first sparsification as the default search pattern.
   - Taylor-weighted, Taylor-fallback, last-step-only, penultimate-step-only, layer-only, or module-only candidates may be used only as explicit controls, side probes, or temporary rescue simplifications after a promising broader family proves too slow.
   - The default frontier for this phase is substantive family replacement, not reducing activation count until the method is almost pure Taylor again.
10. Do not treat scope restriction itself as the main innovation.
   - A candidate whose primary change is a new gate, a tighter usage mask, or a more selective trigger should not be considered a frontier family candidate unless it is paired with a substantively new approximation formula.

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

For retention and next-step planning, use a stricter rule:

- passing the standing baseline means the run is valid
- beating the current champion means the run becomes the new retained best family candidate
- valid-but-not-champion runs should remain in `results.tsv`, but they should not replace the current retained champion as the search anchor

### Metric interpretation

中文说明：

- `sample_seconds` 衡量一次采样实验的运行耗时，越低越好；但本阶段不是单纯追求最快，而是要求候选方法不能明显慢于 standing baseline。
- `sample_seconds_ratio` 是当前耗时相对 standing baseline 的比例，`1.00` 表示与 baseline 基本相同，`1.05` 是本阶段允许的硬上限。
- `latency_within_budget=true` 是质量比较的前置条件；如果延迟超预算，即使图像质量指标更好，也不算本阶段成功。
- `LPIPS` 衡量感知差异，越低表示生成图像越接近 paired baseline/control，是本阶段质量排序的第一优先级。
- `Relative L1` 衡量像素级相对绝对误差，越低越好；它作为 LPIPS 接近时的第二排序指标。
- `SSIM` 衡量结构相似性，越高越好；它用于辅助判断结构是否保持得更好。
- `RMSE` 衡量均方根误差，越低越好；它是最后的 tie-breaker，用于补充像素级误差判断。

How to decide whether a result is good:

1. First check latency: `latency_within_budget` must be true, with `sample_seconds_ratio <= 1.05`.
2. Then compare quality against the standing baseline `20260329-171615`.
3. A candidate is valid only if it is within budget and improves the ordered metric policy: lower `LPIPS`, then lower `Relative L1`, then higher `SSIM`, then lower `RMSE`.
4. A valid candidate is only promoted to the retained family champion if it also beats the current best valid candidate in the same frozen bucket.

Do not interpret the metric numbers as universal absolute quality thresholds.
For this harness, "good" means better than the frozen standing baseline under the same evaluation bucket, and "best" means better than the current retained family champion under that same bucket.

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

中文解释：

- `train.py` 一次只跑一个候选实验，不负责 while 循环。
- “持续搜索”由当前 agent 会话负责：看结果、改函数、再运行下一次。

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
/home/yjs/xdit_env/bin/python train.py   --enable-pade   --pade-m 1   --pade-n 2   --description "candidate: Chebyshev-inspired empirical cubic polynomial under interval-3 latency budget"   > run.log 2>&1
```

Read the summary with:

```bash
grep '^status:\|^paired_outcome:\|^latency_within_budget:\|^sample_seconds:\|^sample_seconds_budget:\|^sample_seconds_ratio:\|^lpips:\|^relative_l1:\|^ssim:\|^rmse:' run.log
```

## Loop

中文解释：

- 每轮循环都必须重新读磁盘上的 `results.tsv` 和最新 `runs/`，而不是依赖聊天上下文记忆。
- 这样即使会话中断，也能从磁盘状态恢复。

LOOP FOREVER until the human stops you:

1. Re-read `results.tsv` and inspect the latest directory under `runs/`.
2. Keep context usage low. Prefer compact summaries from `results.tsv`, `train_summary.json`, `eval_metrics.json`, and `approx_stats.json`.
3. Treat `20260329-171615` as the standing baseline.
4. Identify the current retained family champion:
   - same frozen bucket
   - `enable_pade=true`
   - `latency_within_budget=true`
   - `paired_outcome=better_than_paired_control`
   - best by the metric policy in this file
5. Choose exactly one next polynomial-family hypothesis for `pade_formula_mn()`.
6. Keep diffs small enough that failures are attributable.
7. Commit the change when useful for traceability.
8. Run one experiment.
9. Read the summary from `run.log`.
10. If the run crashed, inspect the traceback, log it, fix the obvious implementation issue, and continue.
11. Use the result to decide the next `pade_formula_mn()` candidate.
   - If it is valid but not better than the retained champion, log it as a non-champion result and continue searching from the retained champion.
   - Do not let a merely baseline-beating run reset the frontier.
12. Continue immediately.

## Search strategy

Suggested order:

1. start with approximation families that can replace TaylorSeer on a meaningful subset of the frozen bucket, not just on a tiny gated corner case
2. prefer one new family hypothesis at a time
3. favor hypotheses that plausibly reduce error through a better formula, basis, or coefficient construction rather than through narrower activation
4. if a candidate is too slow, first simplify the arithmetic inside that family before shrinking it to fewer steps, layers, or modules
5. only after a broader family shows real upside may you introduce limited scope restrictions to recover latency
6. use `pade_step_ratio` and `pade_call_ratio` only as diagnostics, not as optimization targets
7. if a candidate crashes due to an obvious implementation bug, fix the bug and continue the same search direction

## Candidate idea shortlist for `pade_formula_mn()`

To avoid search drift, bias candidate selection toward the following families first.
These are not mandatory, but they are the preferred frontier for this phase.

1. Ordinary empirical cubic / quartic polynomial
   - Treat `factors` and `x` as features, not as a mandate to preserve Taylor coefficients exactly.
   - Prefer formulas that can plausibly run on a broad part of the frozen bucket.
   - Start with the shortest, cheapest form that still differs meaningfully from TaylorSeer.

2. Chebyshev polynomial family
   - Normalize `x` to a bounded interval first.
   - Use a low-order Chebyshev basis for numerical stability.
   - Prefer compact coefficient rules that materially change the approximation, not just its trigger condition.

3. Legendre polynomial family
   - Use a normalized interval and low order.
   - Prefer cases where orthogonal-basis behavior may reduce coefficient interaction noise.

4. Hermite-style cubic family
   - Useful when reusing value and derivative-like information from `factors`.
   - Prefer genuine Hermite-style interpolation behavior rather than Taylor damping with a new label.

5. Low-order rational-polynomial hybrid family
   - Allow cheap rational corrections when they are numerically attributable and not just Padé nostalgia.
   - Prefer hybrids that change the approximation law itself over ones that mostly add safety gates.

6. Piecewise family with a small number of interpretable regimes
   - Different ranges of `x`, derivative patterns, or local coefficient geometry may use different low-order formulas.
   - Prefer a few meaningful regimes over ultra-sparse one-step special cases.

7. Explicit controls
   - Weighted Taylor, cubic damping, sparse gating, last-step-only probes, and Taylor-plus-rational variants are allowed as named controls.
   - They are not the default frontier for this phase.

## Anti-drift guidance for candidate selection

When choosing the next idea, prefer the following order:

1. a family change that can plausibly deliver a material quality gain
2. a new basis or coefficient-construction rule before another coefficient polish on the same family
3. a broader replacement candidate before a narrower gated candidate
4. simplification of a promising but too-slow family before introducing a tighter usage mask
5. numerically safer candidate before more aggressive candidate when both are substantively new
6. latency-budget-compliant candidate before marginally better but slower candidate

Avoid drifting into the following unless the human explicitly asks:

- reopening runtime-parameter search
- redesigning `train.py` beyond bookkeeping or baseline logic
- broad edits across multiple functions in `__init__.py` when `pade_formula_mn()` alone can express the hypothesis
- optimizing Padé identity or theory for its own sake instead of optimizing the practical approximation family
- repeated gate sweeps whose main effect is to use the approximation on fewer steps, layers, or modules
- long runs of tiny coefficient tuning on the same sparse family without introducing a genuinely new approximation mechanism
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
- whether it became the new retained family champion or remained a non-champion valid run
- key metrics: `lpips`, `relative_l1`, `ssim`, `rmse`
- `sample_seconds` and `sample_seconds_ratio`
- the next planned candidate

Immediately after reporting, start the next run unless an allowed blocker is present.
