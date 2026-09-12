# TP4 trace and optimization iteration, 12 September 2026

**Keep the five-token TP4 default.** Seven draft tokens improve repeated coding C1 by 6.2–6.6%, but regress prose by roughly 10–16% and the fresh mixed control at C8 by 6.9%. Two isolated kernel prototypes passed their numerical checks but showed insufficient component gains to justify a claim of 10% better serving performance. No new serving implementation was promoted.

The restored TP8 SG17 reference measured **135.40 coding decode tok/s**, versus **83.21** for the fresh TP4 baseline: **1.63×**. GPU traces explain the scaling: expert GEMM time falls substantially and NVMe staging disappears, while dense computation and communication remain significant. This is a measured comparison between the two configured profiles, not a prediction that doubling tensor parallelism always produces this gain.

## Serving measurements

All throughput below comes from unprofiled requests on a separate client. The original pinned coding and sparkDash workloads, token budgets and timing definitions are unchanged. Each repeat suite contains five C1 coding samples, three C4/C8 coding waves and three prose waves at each concurrency. Every sample is retained in the [machine-readable summary and linked evidence](../results/trace-iteration-01/summary.json).

| Workload, tok/s | Fresh TP4, five tokens | Seven tokens, first suite | Seven tokens, after 299K input | Last suite vs five |
|---|---:|---:|---:|---:|
| Coding C1 decode | 83.21 | 88.68 | 88.37 | +6.2% |
| Coding C4 aggregate | 204.34 | 215.16 | 217.32 | +6.4% |
| Coding C8 aggregate | 332.75 | 340.18 | 331.14 | −0.5% |
| Prose C1 decode | 52.58 | 47.39 | 47.27 | −10.1% |
| Prose C4 aggregate | 117.94 | 103.28 | 103.11 | −12.6% |
| Prose C8 aggregate | 165.91 | 139.66 | 139.20 | −16.1% |

Coding C1 spans 82.34–83.61 for five tokens, 88.50–88.78 for the first seven-token suite and 88.23–88.50 after the long request. These are sequential fixed-configuration runs, not an interleaved full-model A/B experiment. The original publication's two five-token suites remain separately available in [results](results.md); they measured 82.71 and 82.73 coding C1. The new baseline agrees with those runs to within about 0.6% on the mean.

The fresh-prefix control uses the same eight community categories with a new front tag, identical between the two configurations. It is a single C1 and C8 wave per category, not a new quality benchmark or a repeated estimate. Mean category aggregate throughput changes from 55.04 to 54.24 tok/s at C1 and from 213.03 to 198.35 at C8. It supports retaining five tokens for a mixed workload. Complete results: [five](../results/trace-iteration-01/client-holdout-base/holdout/results.json), [seven](../results/trace-iteration-01/client-k7/holdout/results.json).

Coding aggregate timing includes the complete batch duration; sparkDash aggregate decode throughput sums its measured stream decode rates. The two aggregate definitions are preserved from their upstream benchmarks and should not be treated as interchangeable.

## What the TP4 traces show

All four ranks have GPU events for both primary C1/C8 captures. Profiling starts during generation and automatically stops after eight steps. The analysis uses six complete draft-plus-target cycles, excluding the initial cycle and incomplete final boundary. GPU graph correlation IDs identify the boundaries; the expected six draft and eighty target expert GEMMs are checked in each cycle. See [the analyzer and reproduction instructions](../experiments/trace-01/README.md).

The following values are rank-zero means in milliseconds per speculative cycle. Kernel duration sums can overlap across streams; the rows must not be added together or read as mutually exclusive wall-time shares. Communication kernel durations include waiting for peers. Profiled cycle times are diagnostic, not output-token throughput.

| Measurement, ms/cycle | Five, C1 | Seven, C1 | Five, C8 | Seven, C8 |
|---|---:|---:|---:|---:|
| Complete cycle wall time | 60.71 | 67.44 | 103.22 | 127.86 |
| Expert grouped GEMMs | 24.26 | 28.35 | 41.90 | 55.03 |
| B12x dense GEMMs | 10.69 | 10.72 | 11.57 | 12.00 |
| FlashInfer dense GEMMs | 2.95 | 2.90 | 1.88 | 1.80 |
| RoCEnante, including peer waits | 10.73 | 11.95 | 18.15 | 0.00 |
| NCCL, including peer waits | 0.83 | 1.01 | 2.08 | 26.81 |
| Engram staging gaps, GPU idle | 2.15 | 2.57 | 3.82 | 3.09 |

Full per-cycle and all-rank ledgers: [five tokens](../results/trace-iteration-01/traces-baseline/analysis.json), [seven tokens](../results/trace-iteration-01/traces-k7/analysis.json). The primary comparison uses the repeated `baseline-b` and `k7-b` captures. Initial captures are retained too. One initial baseline client assertion ran before scheduler cleanup; generation and trace export had completed. The corrected client waits for idle, and subsequent captures passed.

**NVMe caching:** the graph's ID device-to-host copy and weight/scale host-to-device copies bound two staging intervals per cycle. These intervals include host scheduling and I/O, so they are not pure disk latency. Across ranks, the five-token baseline spends 2.15–2.65 ms there at C1 and 3.00–3.82 ms at C8, with the GPU idle throughout those measured gaps. Eliminating those gaps alone would imply approximately 3–4.6% speedup at fixed work, below a 10% target for these coding traces. A row cache was therefore not implemented in this iteration. Other prompts and longer contexts may have different costs; the traces do not establish a universal cache ceiling.

**Seven-token communication boundary:** the existing RoCEnante path accepts reductions up to 524,288 bytes. At C8, seven draft rows per request produce a 573,440-byte BF16 hidden-state payload; eight target verification rows produce 655,360 bytes. Both exceed the limit and use the existing NCCL fallback. Five-token C8 payloads are 409,600 and 491,520 bytes, which fit. The disappearance of RoCEnante kernels in the seven-token C8 trace confirms that boundary. This is a specific candidate for a subsequent communication experiment, not evidence that increasing the limit is already qualified or fast. It also cannot explain the C1 prose regression, where the larger messages still fit.

## Kernel prototypes

Both tests ran on an otherwise idle fifth Spark while its SG17 service was stopped. They used the same native-head image, original checkpoint partitions and changing synthetic activations. Neither changed the serving model. All timing samples and numerical comparisons are retained with the [source and frozen dependencies](../experiments/trace-01/README.md).

| Prototype | Component result | Decision |
|---|---|---|
| B12x expert path, 96 local experts, full 2,304 intermediate width | 1.016–1.064× at M=6; 1.077–1.078× at M=48 across three routing seeds | Numerical checks passed; too small to establish a 10% serving gain. Not deployed. |
| Zero-pad shared-down K from 576 to 640 at TP4, and 288 to 384 at TP8, enabling the B12x dense path | About 1.45–1.54× at small TP4 M and 1.32–1.49× at small TP8 M; M=48/64 regresses | Promising small component optimization, but only roughly a millisecond of a TP4 cycle. Not deployed. |

The expert test compares against the actual traced FlashInfer `[16,36]` tactic. Its 72 numerical comparisons pass the 0.06 relative-L2 / 0.998 cosine gates; the largest observed relative-L2 error is below 0.01945. These are tolerance-based checks, not bit-identical expert outputs. Cold-L2 CUDA graph measurements use three changing routing seeds with ABBA ordering. [Results](../results/trace-iteration-01/component-results/evidence/result.json).

The padding test preserves checkpoint FP8 bytes and E8M0 scales, adds zero columns and includes activation padding/quantization in the measured graph. All 126 changing-input graph outputs across fourteen TP4/TP8 shapes are bit-identical to the original path on these fixtures; 378 comparisons include both paths against the numerical reference. Larger batches regress, so an unconditional replacement would be inappropriate. Full-model performance and quality were not tested for this prototype. [Results](../results/trace-iteration-01/dense-results/evidence/result.json).

## TP8 reference and extrapolation

SG17 was restored before this reference. Its source identity, native resident Engram path and configuration were checked on all eight ranks. The TP8 reference retains EP4 with MoE TP2, five draft tokens, its 8M logical KV pool and 128 request slots. TP4 uses EP4/MoE TP1, NVMe Engram, a 4M pool and eight slots. Both use the original checkpoint precision and a configured 1M context limit.

Three unprofiled C1 coding samples after an excluded warmup measure **135.14, 136.02 and 135.03 tok/s**, averaging **135.40**. [Client evidence](../results/trace-iteration-01/client-sg17-reference/coding-c1.json). Both C1/C8 traces contain GPU events on all eight ranks.

| Rank-zero measurement, ms/cycle | TP4 C1 | TP8 C1 | TP4 C8 | TP8 C8 |
|---|---:|---:|---:|---:|
| Complete cycle wall time | 60.71 | 39.01 | 103.22 | 79.00 |
| Expert grouped GEMMs | 24.26 | 12.20 | 41.90 | 25.10 |
| Dense GEMMs, B12x + FlashInfer | 13.63 | 10.43 | 13.46 | 9.77 |
| RoCEnante + NCCL, including peer waits | 11.55 | 9.36 | 20.23 | 24.99 |
| Engram staging gaps | 2.15 | 0.00 | 3.82 | 0.00 |

[All-rank TP8 trace ledger](../results/trace-iteration-01/traces-sg17-reference/analysis.json). Zero staging gaps means the native resident path has no NVMe staging sequence; Engram GPU lookup work still exists.

For extrapolation, treat these costs separately. MoE TP2 reduces each rank's expert intermediate width from 2,304 to 1,152, and the measured C1 expert GEMM duration nearly halves. Dense GEMMs fall by only about 24% at C1. At C8, communication duration rises even though expert computation falls. The resulting profiled cycle improvement is about 1.56× at C1 and 1.31× at C8; these are not throughput multipliers, because output tokens accepted per speculative cycle and profiler overhead also matter. The actual unprofiled C1 comparison is 1.63×.

A TP4 optimization transfers to TP8 only to the extent that it improves work still present in TP8. NVMe caching has no direct benefit for SG17's resident path. Small shared-down padding has a measured component benefit at both TP widths, but a limited overall budget. Longer speculation needs a separate TP8 acceptance/performance test; multiplying 135.40 by the TP4 coding gain would ignore workload acceptance, different kernel shapes and communication. Seven-token TP8 serving was not tested.

## Qualification and final state

The seven-token candidate passed text, two C8 arithmetic waves, one/four-image understanding, structured-output and tool checks both before and after the long request. Exact retrieval passed at 32,865, 131,169 and 299,097 input tokens. These are bounded functional checks, not a broad model-quality evaluation or validation of fully occupied 1M context / 4M KV capacity.

The minimum observed OS reserve during the guarded seven-token validation was **19.30 GiB**, above the 2 GiB stop threshold. All four ranks passed final runtime-source, B12x, row-store library, expert-layout and RoCEnante checks without OOMs, restarts or recorded transport faults. The [health receipts](../results/trace-iteration-01/health/final-k7-health.json) and [memory guards](../results/trace-iteration-01/health/tp4-state.json) are retained.

Both TP4 profiles are stopped and preserved. SG17 remains serving on all eight Sparks; [the final post-trace check](../results/trace-iteration-01/health/sg17-final-health.json) passed. Its short reference guard observed a minimum 10.94 GiB OS reserve. Raw traces, complete client results, numerical probes and source hashes are published under [trace-iteration-01](../results/trace-iteration-01/summary.json); the existing five-token runtime remains unchanged.
