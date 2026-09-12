# TP4 trace iteration 01

The [report](../../docs/trace-iteration-01.md) records the decision, measurements and TP8 reference. Five draft tokens remain the selected serving default. The seven-token variant passed the bounded functional tests but regressed prose and mixed workloads. The component prototypes were not deployed.

## Reproduce the seven-token variant

Create a separate checkout from the measured baseline. The patch changes the fixed speculation length and matching readiness/benchmark assertions; all inference overlays and precision settings remain byte-identical.

```bash
git worktree add --detach ../tp4-k7 a92d28383ae7254890f0fcff9c3ebfe8b6c52b14
git -C ../tp4-k7 apply "$PWD/experiments/trace-01/k7.patch"
cp experiments/trace-01/cluster-k7.example.json ../tp4-k7/configs/cluster.local.json
```

Set the local hosts, model/image locations and distinct runtime directory, then follow the existing [qualification and launch instructions](../../docs/reproduction.md) in that checkout. Stop the other GPU service before launching. The patch deliberately preserves the metadata recorded when staging the experiment; [source identity](source-identity.json) and the final results record its qualification and rejection as the default.

Repeat `bench/repeat-client.py`, `validation/acceptance.py` and `validation/long-context.py` as in the baseline instructions. Use fresh output directories and the OS memory guard. The recorded sequence was acceptance, repeat suite, fresh-prefix holdout, traces, 32K/131K/299K retrieval, repeat suite, acceptance.

## Capture and analyze a diagnostic trace

Run the client from a separate machine while the matching server is idle. It starts profiling after each stream has delivered eight nonempty chunks, requests eight profiler steps, saves every response and waits for scheduler cleanup. Chunk counts are not token counts. The server writes one trace on each node; collect those files into `rank0/`, `rank1/`, etc. before analysis.

```bash
python3 experiments/trace-01/trace-client.py \
  --base http://192.0.2.11:8004/v1 --bench bench/v41bench.py \
  --out results/new-profile-client --label unique-profile
python3 experiments/trace-01/analyze-traces.py results/collected-profile
```

This uses SGLang's `/start_profile` API with CPU/GPU activities, no stacks, recorded shapes, detailed annotations and automatic stop. Profiled request timings are diagnostic and must not be used as benchmark throughput. The analyzer uses GPU graph correlation IDs, requires six draft and eighty target expert GEMMs per cycle, and rejects mismatched graphs. Its attribution is specific to this DeepSeek/DSpark execution shape.

The [raw trace manifest](../../results/trace-iteration-01/trace-manifest.json) preserves original and public hashes. Infrastructure metadata is anonymized; every trace event is identical. The first baseline capture completed generation and trace export but its immediate idle assertion raced scheduler cleanup. It remains in the evidence; the corrected client and a repeated capture supplied the primary baseline attribution.

## Component prototypes

`component-results/` compares B12x expert execution with the traced FlashInfer `[16,36]` tactic at TP4's full local expert width. `dense-results/` compares the original shared-down projection with zero padding to B12x's 128-column alignment. Both use original checkpoint weights, synthetic changing inputs, CUDA graph replay and a high-precision numerical reference. Their performance results are component timings, not serving gains.

The included `run-component.py` wrappers reproduce the bounded single-GPU procedure. Their host paths are anonymized and must be set to your local model and dependency directories. Numerical benchmark sources are unchanged; original source hashes are retained with the results. Copy the matching dependency manifest next to each wrapper before running. The full 661-file B12x dependency is frozen in [dependencies](dependencies/identity.json); unpack `b12x-source.tar.gz` into the directory referenced by the wrapper. This dependency is for the prototypes only, and does not replace the serving profile's existing frozen archive.

The runner requires an otherwise idle GPU, verifies the image and every dependency file, uses original model files read-only, and stops its own container if the OS reserve falls below 2 GiB or the deadline expires. The retained benchmark outputs include source hashes, exact checkpoint tensor hashes, numerical checks and all timing samples.
