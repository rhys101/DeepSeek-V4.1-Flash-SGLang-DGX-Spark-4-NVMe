# Qualification scope

All four intended serving Sparks passed the tests below before the full model was launched. These establish implementation correctness for the tested cases; throughput is measured separately.

| Check | Result | Evidence |
|---|---|---|
| Original C++ row-store suite | Five cases per node, including concurrent callers | [CPU reader](../results/qualification/cpu-row-store.json) |
| Packed file reader | Four ownership cases per node, page boundaries and fatal truncation/bounds faults | [CPU reader](../results/qualification/cpu-row-store.json) |
| Actual checkpoint packing | Two files per rank; complete output checksums and 1,037 source-row comparisons per layer | [Packed shards](../results/qualification/packed-engram.json) |
| Upstream four-GPU transport suite | 74 passed and one expected oversized int64 all-gather skip per rank | [Upstream summary](../results/qualification/roce-upstream.json) |
| Actual SGLang transport integration | 24 changing graph replays, mixed RoCEnante/NCCL routes, stable pointers and allocations, scheduler fault rejection on all ranks | [Integration](../results/qualification/roce-integration.json) |
| Actual NVMe GPU lookup | Bit-exact local and TP-assembled checkpoint rows at six eager sizes per layer; 72 changing graph replays per rank | [NVMe GPU](../results/qualification/nvme-gpu.json) |
| Full-model serving | Text, C8, vision, structured JSON, tools and 299K retrieval passed; repeat suite passed afterwards | [Serving results](results.md#serving-validation) |
| Public Dockerfile | Built on Linux ARM64; all eleven baked SGLang source hashes verified | [Build](../results/qualification/build.json) |

The NVMe graph test covers 144, 576 and 1,152 row IDs with twelve changed-input replays per layer and size. It uses the real packed-file callback, real `EngramEmbedding.forward`, original GPU dequantization and an actual four-rank SGLang communication group. The independent reference decodes the original checkpoint bytes on the CPU. The largest eager lookup contains 49,152 row IDs. Empty inputs, invalid checkpoint shapes, ownership boundaries, fixed graph buffers and the absence of a row/scale cache are checked. The probe does not load the complete model.

The transport integration checks eager fallback, large BF16, FP16, MAX reduction, all-gather and non-TP groups, plus eligible captured reductions. Stopping a rank's proxy must poison the transport and make the actual decode and idle scheduler result handlers raise before processing output. It also checks the eventless health path.

Two initial test-harness assertions were corrected and retained in the local experiment record. The upstream result collector expected an `int64` string in pytest's test name, but pytest emits `dtype2`; the original XML was verified without rerunning the successful suite. The integration probe initially retained a 64 KiB assertion from an older test although the serving transport and tested routing use 512 KiB. The corrected integration probe passed on all four GPUs. Neither correction changed the transport algorithm.

The NVMe probe used the same runtime code as serving; a subsequent adapter edit corrected a stale comment about fallback behavior. The original C++ row reader is unchanged from the pinned Mia source. The public build check verifies the alternate packaged base image; the measured model uses the preserved native-head image plus the verified staged runtime kit. See [reproduction](reproduction.md) for that distinction.
