# Comparison targets checked 12 September 2026 UTC

The community benchmark and prompt-set bytes match Tony's latest source exactly. Coding decode measures completion tokens minus one over post-first-token time; coding aggregate includes full batch wall time. The separate sparkDash prose benchmark uses a different prompt, output budget and aggregate decode interval.

| Reference | Precision and hardware | Recorded result | Comparison limit |
|---|---|---|---|
| Mia, TP3 | Original mixed-precision checkpoint; three Sparks; NVMe Engram | Prose C1 37.9 tok/s; C4 aggregate 78.6 tok/s | Different Spark count; her TP4 profile is explicitly not booted |
| Tony, boot 10 | Original mixed-precision checkpoint; four Sparks | Coding C1 73.78 tok/s; C6 coding aggregate 225.48 tok/s | One measured batch per cell; no repeat suite |
| Tony, TP4 EXL3 initial | EXL3 3.5-bit experts; four Sparks | Coding C1 81.54 tok/s | Different expert quantization |
| Tony, TP4 EXL3 repeat 1 / 2 | EXL3 3.5-bit experts; four Sparks | Coding C1 81.20 / 81.11 tok/s; C6 coding aggregate 281.75 / 295.84 tok/s | Later runs on the same process, different expert quantization |
| Our SG17 | Original mixed precision; eight Sparks; resident Engram | Coding C1 134.68 / 134.906 tok/s; repeat C8 aggregate 474.52 tok/s | Different Spark count and Engram storage; not a TP4 result |

Sources: [Mia at e59e6eb](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/tree/e59e6eb67479aa68f6fa700c600dc90a0729b5ec), [Tony at fc725ec](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/tree/fc725ecf10869c184f4347dd73336536d395753c), [Tony's first EXL3 repeat](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/fc725ecf10869c184f4347dd73336536d395753c/results/exl3tp4b-rep1/bench-exl3tp4b-rep1.json), [second repeat](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/fc725ecf10869c184f4347dd73336536d395753c/results/exl3tp4b-rep2/bench-exl3tp4b-rep2.json), [SG17](https://github.com/rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8/blob/66aa6bd5085e63bbb011557075765e3191e807b2/sglang/docs/sg17-rocenante-results.md).

Community runner SHA-256: `e0d6b2d25bd585d11fbdf39c2ddcdf7a4de8ab685af6bd42465e69f3ee6e80a8`. Prompt-set SHA-256: `f8106746dc729d287d3509376d683719862b13f3657847e86153856be6c80bb3`. Retain every complete measurement and report warmup, cache state, timing definition and precision alongside it.
