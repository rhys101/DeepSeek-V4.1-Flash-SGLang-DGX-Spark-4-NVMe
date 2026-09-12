"""Render the fixed TP4/EP4, original-precision NVMe experiment without a GPU."""
import ipaddress
import json
from pathlib import Path
import re


def load(path):
    c = json.loads(Path(path).read_text())
    nodes = sorted(c['workers'], key=lambda node: node['rank'])
    if len(nodes) != 4 or [node['rank'] for node in nodes] != list(range(4)):
        raise ValueError('Exactly four ranks 0..3 are required')
    ips = [str(ipaddress.IPv4Address(node['ip'])) for node in nodes]
    if len(set(ips)) != 4 or ips[0] != c['head_ip']:
        raise ValueError('Unique node addresses and the rank-zero head address are required')
    for node in nodes:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+', node['host']):
            raise ValueError('Invalid SSH destination')
    for key in ['model_store', 'run_dir', 'engram_dir']:
        p = Path(c[key])
        if not p.is_absolute() or str(p) == '/' or '..' in p.parts or any(x in str(p) for x in '\n\r\0,'):
            raise ValueError('Use dedicated absolute paths without traversal or commas')
    sub = Path(c['model_subpath'])
    if sub.is_absolute() or '..' in sub.parts or not sub.parts:
        raise ValueError('Invalid model subdirectory')
    for key in ['name', 'fabric_interface']:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', c[key]):
            raise ValueError('Invalid profile name or interface')
    if not re.fullmatch(r'[=A-Za-z0-9_:,.-]+', c['nccl_hcas']):
        raise ValueError('Invalid HCA selection')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/:@-]*', c['image']):
        raise ValueError('Invalid image reference')
    ipaddress.ip_address(c['api_host'])
    for key in ['dist_port', 'api_port', 'nccl_test_port']:
        if type(c[key]) is not int or not 1024 <= c[key] <= 65535:
            raise ValueError('Invalid port')
    fixed = dict(tp_size=4, ep_size=4, context_length=1000000,
                 max_running_requests=8, max_total_tokens=4000000,
                 chunked_prefill_size=2048, max_prefill_tokens=2048,
                 mem_fraction_static=0.80, draft_length=5,
                 min_free_slots_delay=1, minimum_available_gib=2,
                 io_threads=96, row_cache_gib=0, resident_scales=False)
    for key, value in fixed.items():
        if c.get(key) != value:
            raise ValueError(f'The initial TP4/NVMe profile requires {key}={value!r}')
    if type(c['roce_enabled']) is not bool:
        raise ValueError('roce_enabled must be a boolean')
    c['workers'] = nodes
    return c


def engine_args(c, rank):
    if rank not in range(4):
        raise ValueError('Invalid TP4 rank')
    return [
        '--min-free-slots-delay', '1',
        '--model-path', '/models/' + c['model_subpath'],
        '--served-model-name', 'deepseek-v41-flash',
        '--load-format', 'safetensors', '--dtype', 'bfloat16', '--trust-remote-code',
        '--tp-size', '4', '--ep-size', '4', '--nnodes', '4', '--node-rank', str(rank),
        '--dist-init-addr', f"{c['head_ip']}:{c['dist_port']}",
        '--attention-backend', 'dsv4', '--moe-runner-backend', 'flashinfer_mxfp4',
        '--speculative-moe-runner-backend', 'flashinfer_mxfp4',
        '--fp8-gemm-backend', 'flashinfer_cutlass', '--kv-cache-dtype', 'auto',
        '--context-length', str(c['context_length']),
        '--max-running-requests', str(c['max_running_requests']),
        '--chunked-prefill-size', str(c['chunked_prefill_size']),
        '--max-prefill-tokens', str(c['max_prefill_tokens']),
        '--max-total-tokens', str(c['max_total_tokens']),
        '--mem-fraction-static', str(c['mem_fraction_static']),
        '--cuda-graph-max-bs-decode', '8', '--cuda-graph-bs-decode', *map(str, range(1, 9)),
        '--enable-decoder-swa-bounded-replay', '--disable-custom-all-reduce',
        '--speculative-algorithm', 'DSPARK', '--speculative-dspark-block-size', '5',
        '--speculative-accept-threshold-single', '1.0', '--speculative-accept-threshold-acc', '1.0',
        '--random-seed', '0', '--enable-multimodal',
        '--limit-mm-data-per-request', '{"image":4,"video":0,"audio":0}',
        '--default-chat-template-kwargs', '{"thinking":false}',
        '--tool-call-parser', 'deepseekv41', '--reasoning-parser', 'deepseek-v41',
        '--enable-cache-report', '--watchdog-timeout', '1800',
        '--host', c['api_host'], '--port', str(c['api_port']),
    ]


def environment(c, rank):
    if rank not in range(4):
        raise ValueError('Invalid TP4 rank')
    return {
        'SGLANG4_NVME_PROFILE': '1', 'SGLANG8_RESIDENT_PROFILE': '0',
        'PYTHONPATH': '/opt/sglang4/adapter:/opt/sglang4/runtime:/opt/sglang4/b12x',
        'SGLANG4_ROCE_ALLREDUCE': '1' if c['roce_enabled'] else '0',
        'DSV41_SOURCE': '/models/' + c['model_subpath'],
        'DSV41_PACKED_DIR': '/engram', 'OFFLOAD_MODE': 'nvme',
        'DSV41_CACHE_GIB': '0', 'DSV41_RESIDENT_SCALES': '0',
        'DSV41_IO_THREADS': str(c['io_threads']), 'DSV41_STATS_SECONDS': '60', 'NNODES': '4',
        'SGLANG8_RETRY_FREE_ADMISSION': '0',
        'SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE': '0',
        'SGLANG_ENABLE_DSV41_ENGRAM_KV_PREFETCH': '0',
        'DSV41_TP_PAD': '0', 'DSV41_MXFP8_BACKEND': 'b12x',
        'DSV41_PREFILL_EMPTY_CACHE_TOKENS': '8192',
        'SGLANG_FLASHINFER_MOE_FUSED_FINALIZE': '0',
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:False',
        'SGLANG_RAGGED_VERIFY_MODE': 'static',
        'NCCL_SOCKET_IFNAME': '=' + c['fabric_interface'],
        'GLOO_SOCKET_IFNAME': c['fabric_interface'], 'NCCL_IB_HCA': c['nccl_hcas'],
        'NCCL_IB_DISABLE': '0', 'NCCL_IB_ADDR_FAMILY': 'AF_INET', 'NCCL_NET': 'IB',
        'NCCL_CUMEM_ENABLE': '0', 'NCCL_NVLS_ENABLE': '0',
        'NCCL_BUFFSIZE': '1048576', 'NCCL_LL128_BUFFSIZE': '262144',
        'NCCL_PROTO': '^LL128', 'NCCL_MAX_NCHANNELS': '8',
        'NCCL_DEBUG': 'INFO', 'NCCL_DEBUG_SUBSYS': 'INIT,NET',
        'CUDA_VISIBLE_DEVICES': '0', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'HF_HOME': '/cache/huggingface', 'XDG_CACHE_HOME': '/cache',
        'TRITON_CACHE_DIR': '/cache/triton', 'B12X_ROCE_CACHE_DIR': '/cache/roce',
        'B12X_COMPILE_CACHE_DIR': '/cache/b12x', 'CUTE_DSL_CACHE_DIR': '/cache/cute',
        'FLASHINFER_CUDA_ARCH_LIST': '12.1a', 'TORCH_CUDA_ARCH_LIST': '12.1a', 'MAX_JOBS': '2',
    }
