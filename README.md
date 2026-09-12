# DeepSeek V4.1 Flash on four DGX Sparks: TP4 with NVMe Engram

This separate repository evaluates the original-precision DeepSeek V4.1 Flash checkpoint with SGLang TP4/EP4, exact node-local NVMe Engram lookup and RoCEnante communication. It builds on the [SG17 eight-Spark result](https://github.com/rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8/commit/66aa6bd5085e63bbb011557075765e3191e807b2).

**Status: implementation and qualification in progress. No TP4 throughput is claimed yet.** The eight-Spark result of 134.91 coding decode tokens/s is a reference from a different configuration, not a TP4 measurement.

The initial test profile uses four Sparks, TP4/EP4, native checkpoint MXFP4 expert and FP8 dense weights, BF16 activation dtype, five-token DSpark speculation, a 1M-token context limit, 4M logical KV tokens, eight request slots and 2,048-token prefill chunks. Exact Engram rows remain on each rank's local NVMe, with no additional quantization, row cache or resident scale table. These configured capacities are not yet validated serving capacity.

[Upstream comparison targets](docs/comparison-targets.md) · [Source identity](versions.lock.json)

The row reader has passed the original CPU parity/cache suite, packed ownership and page-boundary cases, concurrent callers, and fatal truncated-file/out-of-range checks on all four nodes. [CPU qualification](results/qualification/cpu-row-store.json). GPU graph and full-model tests remain pending.
