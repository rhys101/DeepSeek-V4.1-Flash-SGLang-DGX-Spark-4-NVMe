# TP4 trace iteration 02

The [report](../../docs/trace-iteration-02.md) includes all repeated workload results, the frozen heterogeneous-arrival evaluation, source changes and limitations. Static five-token serving remains the default. Compact five-token verification and seven tokens with a 1 MiB RoCEnante buffer are optional experiments; the calibration variant is diagnostic only.

## Reproduce a measured variant

Each full patch applies independently to the same published baseline. Create a separate checkout, apply one patch, and supply a local cluster configuration:

```bash
git worktree add --detach ../tp4-compact 1efaec74a8163a204783f73d82bb6d2f4db1bae6
git -C ../tp4-compact apply "$PWD/experiments/trace-02/compact-measured.patch"
cp configs/cluster.example.json ../tp4-compact/configs/cluster.local.json
```

Set real hosts, model/image paths and a distinct name/run directory, then follow the existing [launch and qualification instructions](../../docs/reproduction.md) in that checkout. Stop other GPU services on those four nodes before launch and retain the 2 GiB OS reserve guard. The compact patch keeps `draft_length: 5`. For `k7-roce1m.patch`, use a separate checkout and set `draft_length: 7`. Both patches preserve their original staging metadata; final qualification and selection are recorded in the report and [source identity](source-identity.json).

The compact patch contains the measured cost table and disables SPS recording. Its 14 inference overlays exactly match the final calibration runtime. The serving B12x dependency remains the baseline's frozen dependency; no new kernel dependency is needed. `compact-engine-changes.patch` is a small review diff against the three original engine files in the pinned image, not an additional patch to apply after the complete variant patch.

Run the existing `bench/repeat-client.py`, `validation/acceptance.py` and `validation/long-context.py` with fresh output directories and the memory guard. The measured qualification sequence is acceptance, original repeat suite, holdout, acceptance counters where applicable, traces, 32K/131K/299K retrieval, repeat suite and acceptance again. Profiling is separate from throughput timing.

## Reproduce the frozen workload evaluation

Run from a separate client while the matching server is idle, using this repository's unchanged pinned `bench/v41bench.py`. Copy this experiment directory to the client. The clients assert the TP4 capacity and five-token configuration. Use fresh output directories for both configurations:

```bash
python3 experiments/trace-02/repeat-mixed-client.py \
  --base http://192.0.2.11:8004/v1 --bench bench/v41bench.py \
  --out results/new-mixed --label adaptive
python3 experiments/trace-02/generalization-client.py \
  --base http://192.0.2.11:8004/v1 \
  --out results/new-arrivals --label adaptive
```

Repeat identically on the static baseline, then compare:

```bash
python3 experiments/trace-02/compare-mixed.py \
  results/static-mixed/results.json results/new-mixed/results.json --out results/mixed-comparison.json
python3 experiments/trace-02/compare-generalization.py \
  results/static-arrivals results/new-arrivals --out results/arrival-comparison.json
```

The original fixed workloads, token budgets, prefixes, arrivals, warmups and all three trials are retained. The heterogeneous client uses exact chat-template token IDs and native streaming for per-request verification counters. It measures end-to-end latency and cohort throughput, not the original OpenAI decode metric. Completion status records harness completion separately from exact-answer checks: fact extraction fails strict JSON on both measured profiles. Do not discard this failure or uncapped-output limitations when interpreting throughput.

The [freeze receipt](../../results/trace-iteration-02/frozen-generalization-protocol.json) hashes the measured clients, source manifest, runtime configuration and cost table. Earlier clients are retained in `harness-history/`; the [recovery record](../../results/trace-iteration-02/harness-recoveries.json) explains failed housekeeping assertions and retained data. The timing protocol excludes bounded idle waits.

## Calibration and exact component fixture

To repeat calibration, use a separate checkout with `compact-calibration.patch`. It enables SPS records and supplies an intentionally flat two-point bootstrap table so forced budgets work. Do not report its bootstrap policy as the final adaptive policy. Run `calibrate-client.py --base URL --out FRESH_DIRECTORY`, then `fit-calibration.py FRESH_DIRECTORY/cells.json --out NEW_FIT_DIRECTORY` in Python with NumPy. This runs two orders of 18 real greedy probes plus C8 arithmetic at three budgets. The output fixes 192 tokens for timing consistency and excludes early and draining steps; observer timing calibrates costs only.

The fitting script reproduces `runtime/sps-measured.json` from the retained 36 cells. Request counts are 1/4/8 and token counts span 1–48. The fitted table is static during evaluation. New calibration should be evaluated on separate workloads before any default promotion.

`test-compact-engram.py` runs in the pinned GPU image with the patched candidate checkout mounted read-only at `/candidate`. It imports the original installed Engram implementation as reference and the candidate mappings from that mount. Run on an otherwise idle GPU before installing the candidate overlays; mount the experiment test script and launch it with Python. The fixture compares hashes, request indices, padding, changed graph replay and history commits, with 98 exact comparisons. It is not a full-model quality test.

## Raw traces and package verification

`trace-client.py` accepts `--workload coding` or `--workload prose`, the server's absolute `--server-trace-root /state/trace-iter2`, and an output directory. The server saves traces under `state/trace-iter2` when that root is selected by the caller. Collect each node's output into `rank0/` through `rank3/`, then run:

```bash
python3 experiments/trace-02/analyze-traces.py results/collected-traces
python3 experiments/trace-02/verify-package.py
```

The analyzer checks GPU correlation boundaries and six draft / eighty target expert GEMMs per cycle, retaining six complete cycles per trace. The verifier recomputes all 24 trace attributions, all principal throughput comparisons, source-patch application and byte hashes, the fitted cost table and freeze hashes. It checks document links and final health receipts without launching a GPU service. Python and NumPy are required. Public trace hashes differ from original gzip files because infrastructure metadata is anonymized and gzip is deterministic; every trace event is unchanged. Kernel durations can overlap and communication includes peer waiting, so use unprofiled client results for throughput.
