# Adaptive TP4 speculation and communication, 12 September 2026

**Adaptive verification delivers 5.7% more throughput on a frozen heterogeneous arrival test, but does not establish a general 10% gain.** Mean request latency falls 8.4%; the mean of each trial's median latency falls 11.6%. Repeated category tests exceed 10% on three cases and regress substantially on others. The existing static five-token TP4 profile remains the default; the adaptive implementation is published as an optional experiment with its measured tradeoffs.

The separate seven-token / 1 MiB RoCEnante experiment fixes the communication fallback identified in [iteration 1](trace-iteration-01.md), but still regresses prose. Neither candidate changes checkpoint precision, adds a row cache, or reduces configured context/KV capacity. TP8 adaptive serving was not tested.

## How adaptive is it?

The checkpoint already includes a trained DSpark confidence head. Compact mode uses its scores to choose how much of each request's draft to verify. A host planner estimates the batch budget using confidence information relayed with a two-step lag and a fixed DGX Spark cost table; current GPU scores allocate that budget among request prefixes. Target verification and acceptance rules remain unchanged. There are no prompt-category branches, expected-answer shortcuts or workload-specific length settings. This is per-step adaptation, not an online learner that refits its weights or cost model during service.

The cost table models cycle time as `bias + alpha(request_count) + theta(verification_rows)`. It was fitted once from a separate database-index explanation prompt, at request counts 1, 4 and 8 and six forced budgets, with two measurement orders. All 36 cells are retained. The fit uses 18 trial medians and 16 parameters: the small in-sample residual is not strong generalization evidence. Maximum residual is 4.64%; a leave-one-interior-cell interpolation check reaches 13.93%, with endpoints and structurally necessary overlap cells explicitly excluded. Hardware calibration is appropriate for this DGX Spark deployment; its limited coverage of expert routing and workload-dependent kernel costs is the remaining concern. [Calibration records](../results/trace-iteration-02/client-compact5e-calibration-resume01/calibration/cells.json), [fit](../results/trace-iteration-02/compact5-sps-fit/fit.json), [interpolation check](../results/trace-iteration-02/compact5-sps-fit/interpolation-check.json).

Three engine changes make short verification useful and correct here: finer token-count CUDA graphs, Engram hashing with variable request boundaries, and sparse-attention request mapping for compact rows. Both mappings handle graph padding and replayed boundaries. The existing graph grid otherwise rounds a one-request short verification back up to six rows. The new grid has 21 sizes from 1 to 48 rows. [Review-sized engine diff](../experiments/trace-02/compact-engine-changes.patch); [complete staged-source patches and hashes](../experiments/trace-02/source-identity.json).

On the frozen arrival test, the fiction request averages 2.29 verification rows per step, interval-merging code 4.35, and arithmetic 6.00, versus six throughout the static baseline. These counts include the mandatory bonus row. They demonstrate that the policy reacts to requests inside a changing batch. They do not by themselves prove a speedup. [All request counters](../results/trace-iteration-02/generalization-comparison.json).

## Frozen heterogeneous arrivals

Twelve previously unused tasks arrive at fixed offsets over 6.4 seconds: fiction, two coding tasks, arithmetic, incident summary, sorting, translation, concurrency explanation, requirements, fact extraction, service design and editing. Each configuration gets one excluded warmup cohort and three measured trials. The comparison asserts identical protocol and tokenized inputs. Inference source and the cost table were frozen before this evaluation and were not retuned using its outcomes. [Protocol and freeze hashes](../results/trace-iteration-02/frozen-generalization-protocol.json).

| Metric, mean of three trials | Static five | Adaptive five | Change |
|---|---:|---:|---:|
| Cohort output throughput, tok/s | 100.44 | 106.16 | +5.7% |
| Cohort wall time, seconds | 22.92 | 21.69 | −5.3% |
| Mean request latency, seconds | 10.22 | 9.36 | −8.4% |
| Median request latency, seconds | 10.83 | 9.58 | −11.6% |
| Output tokens per cohort | 2,301.7 | 2,303.0 | +0.06% |

Static cohort throughput is 99.75, 101.44 and 100.13 tok/s; adaptive is 107.09, 105.86 and 105.53. Every request type has lower mean latency in these three trials. This native-streaming metric includes prefill, scheduling and the arrival pattern; it is distinct from the original OpenAI decode benchmark. The runs are sequential, not interleaved independent deployments, and three repeated cohorts are a small sample. Full responses, individual timings and excluded warmups are retained for [static](../results/trace-iteration-02/client-baseline-generalization-v2/results.json) and [adaptive](../results/trace-iteration-02/client-compact5-generalization-v2/results.json).

Arithmetic and sorted-word exact checks pass on both configurations in all three trials. Fact extraction fails strict JSON on both: each returns the correct values inside Markdown fences. That remains a failure; the evaluator was not relaxed. Many open-ended responses reach their fixed output caps. This is a bounded functional check, not a comprehensive quality benchmark, and full-model outputs are not claimed bit-identical.

## All repeated community categories

The pinned eight-category suite uses identical new prefix tags and original output budgets, with an excluded warmup and three measured waves at C1 and C8. The final trial reverses case order. C1 below is per-stream decode throughput; C8 is batch aggregate throughput including time to first token. All 48 measured batches per configuration are retained. [Complete comparison, samples and standard deviations](../results/trace-iteration-02/mixed-repeat-comparison.json).

| Workload | C1 tok/s, static → adaptive | C8 tok/s, static → adaptive |
|---|---:|---:|
| Coding | 78.89 → 79.05 (+0.2%) | 349.41 → 303.25 (-13.2%) |
| Json | 67.38 → 48.33 (-28.3%) | 193.62 → 178.77 (-7.7%) |
| Narrative | 33.73 → 36.51 (+8.3%) | 107.76 → 128.46 (+19.2%) |
| Prose | 36.57 → 42.70 (+16.7%) | 131.04 → 141.39 (+7.9%) |
| Math | 83.56 → 78.71 (-5.8%) | 349.03 → 286.02 (-18.1%) |
| Reasoning | 68.17 → 65.72 (-3.6%) | 248.47 → 218.72 (-12.0%) |
| Summary | 34.32 → 35.91 (+4.6%) | 122.76 → 141.09 (+14.9%) |
| Format | 88.25 → 87.73 (-0.6%) | 296.90 → 308.23 (+3.8%) |

Narrative C8, prose C1 and summary C8 clear 10% in this repeated fixture. JSON C1, math C8 and coding C8 regress markedly. A selected-category average would conceal those losses. Earlier single-wave holdouts remain in the evidence but do not replace these repeats.

The original coding and sparkDash controls also reject a general default change. Against the nearby restored static baseline, adaptive coding changes +3.1% / −7.4% / −9.8% at C1/C4/C8; sparkDash prose changes −8.4% / +1.4% / +7.8%. Static coding C1 is stable across the two baseline brackets, 83.06 and 82.96 tok/s. The first comparison gives the same directional result. [Nearby control](../results/trace-iteration-02/nearby-baseline-comparison.json), [earlier control](../results/trace-iteration-02/compact5-measured-after-vs-baseline.json). Coding aggregate and sparkDash aggregate retain their different upstream timing definitions.

## Seven tokens with a larger communication buffer

Raising the RoCEnante capacity from 512 KiB to 1 MiB keeps seven-token C8 hidden-state reductions on RoCEnante. Each of eight target graphs on rank zero changes from 83 NCCL all-reduces and zero RoCEnante kernels to zero NCCL all-reduces and 83 RoCEnante kernels. Five-token baseline graphs contain two NCCL reductions and 81 RoCEnante kernels. [Raw-trace count audit](../results/trace-iteration-02/roce-capacity-completed.json).

After bounded long-context validation, seven tokens / 1 MiB measures 88.64 coding C1 tok/s (+6.7% versus the fresh five-token baseline), 216.15 aggregate C4 (+4.2%) and 338.33 aggregate C8 (+2.4%). Prose changes −9.8%, −12.6% and −13.1%. Fixing the fallback is real, but does not make longer speculation a broad win. [Full paired suites](../results/trace-iteration-02/k7roce1m-after-vs-baseline.json).

## Traces and remaining bottleneck

This iteration adds 24 raw GPU traces: baseline prose, seven-token / 1 MiB coding, and adaptive coding, each at C1/C8 on all four ranks. Every trace contains GPU events and six complete analyzed speculative cycles. All trace events and numerical values are preserved; only infrastructure metadata is anonymized. [Manifest](../results/trace-iteration-02/trace-manifest.json), [reproduction instructions](../experiments/trace-02/README.md).

In the adaptive coding C8 trace, rank-zero cycle time is 117.40 ms and expert GEMMs total 53.49 ms, versus 103.22 and 41.90 ms in the prior static coding trace. Engram staging gaps are 3.78 versus 3.82 ms. Fewer verified rows do not guarantee faster expert kernels. These are diagnostic, separately captured timings: kernel durations can overlap, communication includes peer waits, and profiler timing is not serving throughput. The evidence points toward a cost model that accounts for expert execution and graph overhead; it does not establish the cause of every regression. [Adaptive ledger](../results/trace-iteration-02/traces-compact5-measured/analysis.json), [static coding ledger](../results/trace-iteration-01/traces-baseline/analysis.json).

## Qualification, failed attempts and final state

Both final candidates passed text, C8 arithmetic, one/four-image understanding, structured-output and tool checks before and after long-context testing. Exact retrieval passed at 32,869, 131,173 and 299,101 input tokens. TP4 remains TP4/EP4/MoE-TP1 with original MXFP4 experts, FP8 dense weights, BF16 activations, 1M configured context, a 4M logical KV pool, eight request slots, 96 I/O threads, no row cache and no resident scale table. Full 1M context or 4M occupied KV capacity was not stress-tested.

The final compact component fixture passes 98 exact comparisons covering hash rows, sparse request indices, changing CUDA graph replay and history commits. These component checks do not imply bit-identical generated text. The serving guards observed minima of 19.29 GiB for the adaptive candidate and 19.08 GiB for seven tokens / 1 MiB, above the 2 GiB stop threshold. Final four-rank source, configuration and transport checks passed without OOMs or restarts during these qualified runs. [Component checks](../results/trace-iteration-02/compact-layout-component-test.json), [adaptive health](../results/trace-iteration-02/health/final-compact5-measured-health.json), [seven-token health](../results/trace-iteration-02/health/final-k7roce1m-health.json), [guard histories](../results/trace-iteration-02/health/final-state.json).

Earlier compact prototypes failed capture on uniform-layout assumptions, or were stopped after source audits found padding/replay issues. They produced no qualifying benchmark result; their state history is retained. A calibration client initially misread the successful `[true]` control response. Later evaluation clients hit an immediate-idle race: all 48 mixed timings were retained, while the incomplete first heterogeneous cohort was preserved as a failed harness attempt. The corrected client saves each response and waits for housekeeping outside timing; both runtimes use the same corrected protocol and excluded warmup. A final baseline health collector initially included shutdown tracebacks from an earlier boot; scoping logs to the current start yielded a clean run. These corrections changed no inference policy. [Attempt history](../results/trace-iteration-02/health/compact-attempts-state.json), [harness recoveries](../results/trace-iteration-02/harness-recoveries.json).

All TP4 experiment containers are stopped and preserved. The original SG17 service is restored on all eight Sparks, with its source and health checked. [Restoration receipt](../results/trace-iteration-02/health/sg17-restored-health.json). The published default runtime is unchanged. [Package verification](../results/trace-iteration-02/verification.json).
