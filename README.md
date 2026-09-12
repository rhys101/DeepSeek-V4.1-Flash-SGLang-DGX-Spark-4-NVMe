# DeepSeek V4.1 Flash on four DGX Sparks: TP4 with NVMe Engram

This separate repository evaluates the original-precision DeepSeek V4.1 Flash checkpoint with SGLang TP4/EP4, exact node-local NVMe Engram lookup and RoCEnante communication. It builds on the [SG17 eight-Spark result](https://github.com/rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8/commit/66aa6bd5085e63bbb011557075765e3191e807b2).

**Status: four-rank serving and capability smoke checks passed; benchmark collection is in progress. No final TP4 throughput is claimed yet.** The eight-Spark result of 134.91 coding decode tokens/s is a reference from a different configuration, not a TP4 measurement.

The initial test profile uses four Sparks, TP4/EP4, native checkpoint MXFP4 expert and FP8 dense weights, BF16 activation dtype, five-token DSpark speculation, a 1M-token context limit, 4M logical KV tokens, eight request slots and 2,048-token prefill chunks. Exact Engram rows remain on each rank's local NVMe, with no additional quantization, row cache or resident scale table. The server resolves these settings; full-pool and 1M-input occupancy have not been validated.

[Upstream comparison targets](docs/comparison-targets.md) · [Qualification](docs/qualification.md) · [Reproduction](docs/reproduction.md) · [Source identity](versions.lock.json)

The row reader has passed the original CPU parity/cache suite, packed ownership and page-boundary cases, concurrent callers, and fatal truncated-file/out-of-range checks on all four nodes. The real NVMe GPU lookup and changing CUDA graphs are bit-exact against checkpoint bytes on all four ranks. The upstream transport suite and actual SGLang graph/fault integration also passed. [Qualification details](docs/qualification.md). Full-model startup and the text, C8, vision, structured JSON and tool-call smoke checks passed. Throughput measurement is in progress.
