# Reproducing the TP4 NVMe experiment

Use four Linux ARM64 DGX Sparks with the same checkpoint on node-local NVMe and a working RoCE fabric. The public example uses documentation addresses. Keep private settings in the ignored `configs/cluster.local.json`.

## Build the pinned base

Run on a Spark from the repository root:

```bash
docker buildx build --load --platform linux/arm64 -f docker/Dockerfile \
  -t deepseek-v41-sglang4:tp4-nvme .
docker image inspect --format '{{.Id}}' deepseek-v41-sglang4:tp4-nvme
cp configs/cluster.example.json configs/cluster.local.json
```

Set `image` and `expected_image_id` to that tag and ID. Set the model directories, a fresh `run_dir`, a persistent node-local `engram_dir`, the four worker addresses and fabric settings. Distribute the same image to each worker with `docker save` / `docker load`, and verify its image ID on every worker. The Docker build verifies all eleven SGLang overlays against the pinned base before installing them. Staging subsequently verifies every base source against either the pinned original or the exact final overlay.

The measured experiment reuses the already installed SG17 native-head image and mounts the eleven verified overlays. A separate build check exercises this repository's Dockerfile. A successful image build alone is not a complete fresh-install serving reproduction.

## Stage and pack

The controller runs on rank zero. It uses passwordless SSH over the configured fabric for the other three ranks and refuses to launch while another GPU container or compute process is active.

```bash
python3 scripts/cluster.py dry-run
python3 scripts/cluster.py stage
python3 scripts/cluster.py pack
```

Staging creates a fresh runtime kit, extracts and verifies all 241 frozen B12x files, and compiles the unchanged C++ row store inside the pinned image on every worker. All four compiled-library hashes must agree.

Packing writes approximately 47.21 GiB on each worker. Each packed row retains the original 256 FP8 weight bytes and eight E8M0 scale bytes. The packer records complete output SHA-256 checksums and compares 1,037 deterministic random and boundary rows per layer against the original checkpoint. Existing files require valid manifests and matching checksums. Model weights and packed shards are not distributed in this repository.

## Qualify and serve

```bash
python3 scripts/qualify-gpu.py --config configs/cluster.local.json --suite roce-upstream
python3 scripts/qualify-gpu.py --config configs/cluster.local.json --suite roce-integration
python3 scripts/qualify-gpu.py --config configs/cluster.local.json --suite nvme
python3 scripts/cluster.py serve
```

The upstream suite has exactly one expected skip: an int64 all-gather shard exceeds the test runtime's buffer capacity. The integration probe exercises the actual SGLang communication route and scheduler fault checks. The NVMe probe compares actual checkpoint lookups and changing graph inputs with independently decoded CPU bytes.

`serve` validates the source composition, TP4/EP4 configuration, target and draft expert shapes, required NVMe storage and RoCEnante activation on every rank. Its startup guard stops the experiment on a failed container, OOM, restart or less than 2 GiB of available OS memory. `stop` stops only the named four-rank experiment. `start` restarts those preserved containers.

## Measure from a separate client

Use Python 3.10 or newer and Node 22.19 or newer. Copy `bench/` and `validation/` to the client and install the locked Node dependency:

```bash
npm ci --ignore-scripts --prefix bench/sparkdash
bash bench/run-community.sh http://192.0.2.11:8004/v1 results/new-community tp4-nvme
python3 bench/repeat-client.py --base http://192.0.2.11:8004/v1 \
  --out results/new-repeat --label tp4-nvme
python3 validation/acceptance.py --base http://192.0.2.11:8004/v1 --out results/new-acceptance
python3 validation/long-context.py --base http://192.0.2.11:8004/v1 \
  --out results/new-long-context --tag unique-trial-tag
```

Run the OS-only guard on rank zero during the benchmark:

```bash
python3 scripts/memory-guard.py --config configs/cluster.local.json --label benchmark
```

Signal completion by creating `state/benchmark-guard-stop` under the configured runtime directory. Use a new label on the next run. Do not issue GPU telemetry queries while timing requests. Retain the guard receipt, resolved server configuration, source manifests, complete benchmark responses and every repeat.

The configured 1M context limit and 4M logical KV pool are settings. The bundled long-context smoke tests reach approximately 299K input tokens and do not validate a fully occupied 4M pool or broad long-context quality.
