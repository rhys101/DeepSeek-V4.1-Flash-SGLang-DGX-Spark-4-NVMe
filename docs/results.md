# TP4 with NVMe Engram: measured results

Four Sparks delivered **82.73 coding decode tokens/s at C1** and **340.41 aggregate coding tokens/s at C8** in the final repeat suite, using the original mixed-precision checkpoint with exact NVMe Engram lookup. C1 coding was stable across the two suites: 82.708 before the dedicated long-context probes and 82.730 afterwards.

The comparison with the latest Tony EXL3 profile is mixed. Coding C1 is slightly higher, coding C6 lies within his published range, and mixed-workload C6 is higher; mixed-workload C1 is lower than his latest repeats. This is a comparison with published measurements on another cluster, not a controlled claim that one implementation universally wins. Tony's EXL3 experts also use different quantization.

## Repeated workloads

| Metric, tok/s | After community sweep | After 299K retrieval |
|---|---:|---:|
| Coding C1 decode | 82.71 | 82.73 |
| Coding C1 aggregate | 75.99 | 75.65 |
| Coding C4 aggregate | 202.38 | 211.81 |
| Coding C8 aggregate | 335.47 | 340.41 |
| Prose C1 decode | 52.61 | 52.62 |
| Prose C4 aggregate | 117.02 | 116.91 |
| Prose C8 aggregate | 162.49 | 162.18 |

Each suite contains five C1 coding samples and three C4/C8 coding waves. The C1 ranges were 82.44–82.89 and 82.53–82.87 tok/s, respectively. Each prose result averages three waves. Every coding request completed 200 tokens; every prose request completed 256. The two C8 coding means average six complete waves in total; 340.41 is the mean of the final three, not the fastest individual wave.

Raw evidence: [first coding suite](../results/runs/repeat-fresh/coding.json), [first prose suite](../results/runs/repeat-fresh/sparkdash.json), [final coding suite](../results/runs/repeat-after-long/coding.json), [final prose suite](../results/runs/repeat-after-long/sparkdash.json), [summary with ranges and sample deviations](../results/measurement-summary.json).

## Complete community sweep

The sweep contains 63 category/concurrency batches and 261 completed requests. The eight-category average excludes the separate counting ceiling probe. Each cell below is one measured batch or the stated category average; it is not a repeat-suite mean.

| Concurrency | Coding decode per stream | Coding aggregate | Eight-category aggregate | Eight-category decode per stream |
|---|---:|---:|---:|---:|
| C1 | 82.99 | 76.38 | 52.04 | 58.25 |
| C2 | 71.87 | 117.31 | 87.25 | 50.46 |
| C3 | 62.71 | 158.47 | 111.99 | 42.94 |
| C4 | 56.25 | 206.90 | 139.89 | 40.29 |
| C5 | 51.67 | 238.08 | 160.68 | 37.25 |
| C6 | 52.62 | 283.36 | 180.60 | 35.45 |
| C8 | 46.32 | 330.81 | 212.76 | 31.44 |

[Complete community request timings, usage counts and prefill results](../results/runs/community.json).

## Published comparison targets

All rates below are tokens/s. The Tony EXL3 repeat columns are the later runs published for the same four-Spark profile. Upstream revisions were rechecked on 12 September 2026 UTC.

| Matched metric | This TP4/NVMe experiment | Tony EXL3 repeat 1 | Tony EXL3 repeat 2 |
|---|---:|---:|---:|
| Coding C1 decode | 82.73, final five-sample mean | 81.20 | 81.11 |
| Coding C6 aggregate | 283.36, community sweep | 281.75 | 295.84 |
| Eight-category C1 aggregate | 52.04 | 54.11 | 57.15 |
| Eight-category C6 aggregate | 180.60 | 170.92 | 176.14 |

Sources: [Tony repeat 1](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/fc725ecf10869c184f4347dd73336536d395753c/results/exl3tp4b-rep1/bench-exl3tp4b-rep1.json), [repeat 2](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/fc725ecf10869c184f4347dd73336536d395753c/results/exl3tp4b-rep2/bench-exl3tp4b-rep2.json). His [initial EXL3 run](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/fc725ecf10869c184f4347dd73336536d395753c/results/exl3tp4b/bench-exl3tp4b.json) recorded coding C1 81.54 and C6 aggregate 292.07, so our C1 is about 1.5–2.0% above his three published samples. That small cross-cluster difference is best described as competitive. His native-checkpoint [boot 10](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/fc725ecf10869c184f4347dd73336536d395753c/results/boot10/bench-boot10.json) recorded coding C1 73.78 and C6 aggregate 225.48.

On the separate sparkDash prose workload, our final C1 mean is 52.62 versus [Mia's reported TP3 37.9](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/blob/e59e6eb67479aa68f6fa700c600dc90a0729b5ec/README.md), about 38.8% higher; our C4 aggregate is 116.91 versus 78.6. These use different Spark counts. Mia explicitly labels her TP4 profile unbooted, so there is no measured Mia TP4 result to beat. Her exact source version for the reported prose figures is not established; our sparkDash source is pinned and verified.

[Pinned target data and original upstream JSON digests](../results/reference/published-targets.json). The earlier [eight-Spark SG17 result](https://github.com/rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8/blob/66aa6bd5085e63bbb011557075765e3191e807b2/sglang/docs/sg17-rocenante-results.md) remains a different configuration: eight Sparks, resident Engram, an 8M logical pool and 128 request slots.

## Method and limits

The unchanged community runner has SHA-256 `e0d6b2d25bd585d11fbdf39c2ddcdf7a4de8ab685af6bd42465e69f3ee6e80a8`, matching Tony's current runner. Its prompt-set file has SHA-256 `f8106746dc729d287d3509376d683719862b13f3657847e86153856be6c80bb3`. Coding decode uses `(completion_tokens - 1) / (total_time - TTFT)`. Coding aggregate uses total completion tokens over complete batch wall time, including startup/TTFT. The sparkDash prose aggregate uses its post-first-token decode interval; those aggregate definitions must not be mixed.

Requests ran from a separate client with Python 3.12.3 and Node 22.22.2. Thinking was disabled, temperature was zero, and usage tokens were required. No GPU telemetry queries ran during timed workloads. The only cluster sampler read OS memory and container health.

The community sweep ran after capability smoke tests, using the original warmup and deterministic tags. The repeat suites use additional excluded coding warmups, identical reused prefixes, five C1 samples, three C4/C8 waves, and reversed coding concurrency order on trial two. The internal `repeat-fresh` directory names the first repeat suite, not a cold-cache run: the server had already completed the community sweep and prefill requests up to 93,335 actual input tokens. Neither repeat flushes the cache. All warmups and measured batches are retained.

The profile is TP4/EP4 with MoE TP1, 96 target and 32 draft experts per rank, a 2,304-wide local intermediate, DSpark block size five, eight request slots and 2,048-token prefill chunks. Engram storage uses about 47.21 GiB per node with exact original FP8 weight and E8M0 scale bytes, O_DIRECT packed reads, 96 I/O threads, no row cache and no resident scale table. There is no additional weight quantization. No separate NCCL-only or resident-Engram ablation was run for this TP4 result.

## Serving validation

The text, C8 arithmetic, one/four-image, structured-JSON and tool-round-trip checks passed before and after the long-context probes. Three-record retrieval passed at 32,870, 131,174 and 299,102 actual prompt tokens, taking 10.01, 54.52 and 164.10 seconds end to end. This is a bounded repeated-filler retrieval smoke test, not a broad quality evaluation. The configured 1M context limit and complete 4M logical KV pool occupancy were not stress-tested.

The minimum observed OS memory reserve was 24.90 GiB during startup and 19.52 GiB during serving validation, above the 2 GiB guard floor. The largest measured swap increase during validation was 3.21 MiB on one rank; the other ranks showed no increase. All four ranks finished running without OOMs or restarts, with zero recorded RoCEnante transport errors. [Guard receipt](../results/serving/validation-guard.json), [long-context responses](../results/serving/long-context/result.json), [capabilities before](../results/serving/acceptance/result.json), [capabilities after](../results/serving/acceptance-after/result.json), [final health and resolved configuration](../results/serving/final-health.json).

All four ranks had identical active SGLang, adapter, runtime, frozen B12x and compiled row-store identities. The measured runtime matches source commit `25e7cb82f6524e3bbc46db6a9a54125e75e36d0e`; the later lock-file update records completed measurement status. Checkpoint config/index and packed Engram digests were recorded; complete hashes of all original checkpoint weight files were not computed. The public Docker build passed independently, while measured serving reused the native-head image with verified mounts. [Source identity](../results/source-identity.json), [qualification](qualification.md), [reproduction](reproduction.md).

After measurement, the TP4 containers were stopped and preserved. The original SG17 service was restored on all eight Sparks and passed its source, layout, resident-Engram and transport health checks. [Restoration receipt](../results/serving/restored-sg17.json).
