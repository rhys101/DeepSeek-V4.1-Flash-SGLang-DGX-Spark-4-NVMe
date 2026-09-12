# DeepSeek V4.1 Flash on four DGX Sparks: TP4 with NVMe Engram

**82.73 coding decode tok/s at C1 · 340.41 aggregate coding tok/s at C8**, retaining the original checkpoint precision. These are repeat-suite means after a successful 299,102-token retrieval check, measured on 12 September 2026.

A subsequent [trace and optimization iteration](docs/trace-iteration-01.md) retained the five-token default: seven tokens improved coding C1 by about 6% but regressed prose. The same iteration measured the restored TP8 reference at 135.40 coding tok/s and published all-rank GPU traces, explaining the observed 1.63× C1 scaling.

A second [adaptive-verification iteration](docs/trace-iteration-02.md) gained **5.7% throughput on a frozen heterogeneous arrival test**, with 8.4% lower mean request latency. Some repeated categories gained over 10%, while others regressed substantially. The complete optional patches and evidence are published; static five-token serving remains the default.

This separate repository contains the SGLang TP4/EP4 implementation, exact node-local NVMe Engram storage, four-rank RoCEnante integration, qualification evidence and complete benchmark results. The [SG17 eight-Spark project](https://github.com/rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8) remains separate.

| Workload, tok/s | First repeat suite | Repeat after 299K input |
|---|---:|---:|
| Coding C1 decode | 82.71 | 82.73 |
| Coding C4 aggregate | 202.38 | 211.81 |
| Coding C8 aggregate | 335.47 | 340.41 |
| Prose C1 decode | 52.61 | 52.62 |
| Prose C4 aggregate | 117.02 | 116.91 |
| Prose C8 aggregate | 162.49 | 162.18 |

Each suite contains five C1 coding samples, three C4/C8 coding waves and three prose waves per concurrency. C1 coding ranges were 82.44–82.89 in the first suite; all samples and both suites are retained. Coding uses the unchanged 200-token community workload; prose uses the pinned 256-token sparkDash workload. Aggregate coding timing includes the complete batch duration. [Full results and timing definitions](docs/results.md).

Against the latest published targets, this profile is competitive with Tony's TP4 EXL3 configuration: coding C1 is slightly higher, coding C6 sits within his published range, mixed-workload C6 is higher, and mixed-workload C1 is lower than his latest repeats. His EXL3 experts use different quantization. Prose C1 reaches 52.62 tok/s versus Mia's reported 37.9 on **three** Sparks; her four-Spark profile is documented but unbooted. [Comparison targets and pinned sources](docs/comparison-targets.md).

The tested profile uses four Sparks, TP4/EP4, native MXFP4 expert and FP8 dense checkpoint weights, BF16 activations, five-token DSpark speculation, eight request slots, 2,048-token prefill chunks, a configured 1M context limit and a 4M logical KV pool. Each rank stores about 47.21 GiB of exact packed Engram rows on local NVMe, with **no row cache, resident scale table or additional quantization**. The 1M limit and full 4M pool occupancy have not been stress-tested; bounded retrieval passed at 32,870, 131,174 and 299,102 input tokens.

Text, C8 arithmetic, one/four-image understanding, structured JSON and a tool round trip passed before and after the long request. Both Engram tables were bit-exact against original checkpoint bytes in the GPU lookup and changing-graph tests. The minimum observed OS memory reserve during serving validation was **19.52 GiB**; all four ranks finished healthy without OOMs, restarts or recorded RoCEnante transport faults.

[Reproduction](docs/reproduction.md) · [Qualification](docs/qualification.md) · [Source identity](results/source-identity.json) · [Machine-readable summary](results/measurement-summary.json) · [Sanitized measured profile](profiles/measured.example.json)
