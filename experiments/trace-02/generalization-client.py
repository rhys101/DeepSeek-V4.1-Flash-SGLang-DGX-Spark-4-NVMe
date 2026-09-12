"""Frozen heterogeneous arrivals; native streaming supplies exact acceptance counters."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path

CASES = [
    dict(id='scene', at=0.0, budget=320, prompt='Write a quiet scene in which a night-shift train dispatcher finds a handwritten note inside an old timetable. Reveal the significance through action and dialogue. Avoid a twist ending.'),
    dict(id='interval_code', at=0.0, budget=320, prompt='Implement a Python function merge_intervals(intervals) that merges overlapping closed integer intervals. Return a sorted list, accept empty input, and do not mutate the input. Explain the invariant and give three edge cases.'),
    dict(id='arithmetic', at=0.0, budget=64, prompt='Return only a JSON object with keys product, square_sum, and gcd. product is 37 times 29; square_sum is the sum of the squares of the integers 1 through 12 inclusive; gcd is the greatest common divisor of 84 and 126.', expected={'product':1073, 'square_sum':650, 'gcd':42}),
    dict(id='incident_summary', at=0.5, budget=256, prompt='Summarize these incident notes for the next on-call engineer in six concise bullets. Distinguish observations from hypotheses. At 09:12 the West region checkout error rate rose from 0.3% to 4.8%. Read-only pages stayed healthy. At 09:18 the team disabled the new address-validation rule, but errors persisted. At 09:24 database connection wait time was 2.1 seconds, while query execution remained under 40 milliseconds. At 09:31 a worker retry limit was reduced from ten attempts to two. Error rate returned to 0.5% by 09:39. No data loss has been found. The team suspects retry amplification but has not established the initial trigger. The next engineer should inspect connection-pool limits, retain request traces, and compare the affected region with East.'),
    dict(id='sort_json', at=1.0, budget=64, prompt='Return only JSON with key ordered, whose value is these words sorted alphabetically: walnut, ash, birch, elm, cedar.', expected={'ordered':['ash','birch','cedar','elm','walnut']}),
    dict(id='translation', at=1.6, budget=160, prompt='Translate this note into natural French, preserving its polite but direct tone: We can deliver the first part on Thursday. The remaining items need one more inspection, so please keep Friday morning available. If that causes a problem, let us know today and we will arrange another time.'),
    dict(id='concurrency_explanation', at=2.2, budget=320, prompt='Explain a lost-update race using two workers incrementing the same counter. Show a short interleaving, then explain how a mutex and an atomic increment each prevent the error. Include one limitation of each approach.'),
    dict(id='requirements', at=2.8, budget=256, prompt='Turn this request into a concise implementation checklist with acceptance criteria: A library needs a reservation system. Members may reserve up to three books, but reference books cannot be reserved. A returned reserved book is held for 48 hours. The first eligible member in the queue gets notified, and a missed pickup moves the book to the next member. Staff can override a hold, but every override needs a reason and an audit record. Avoid inventing extra features.'),
    dict(id='extract_facts', at=3.7, budget=96, prompt='Extract only the final approved values as JSON with keys owner, date, rooms. Notes: On Monday, Priya proposed 12 rooms for 14 October. On Tuesday, Omar suggested 10 rooms for 16 October. The final review assigned ownership to Lina, approved 11 rooms, and set the date to 18 October. The earlier proposals were withdrawn.', expected={'owner':'Lina','date':'18 October','rooms':11}),
    dict(id='technical_design', at=4.5, budget=384, prompt='Design a small service that sends appointment reminders. Explain how it handles retries, duplicate delivery, time zones, cancellation, and a worker crash after sending but before recording success. State the delivery guarantee you can honestly provide and describe a practical test for the crash case.'),
    dict(id='edit_note', at=5.4, budget=160, prompt='Rewrite this message to be clear and calm, keeping every commitment and avoiding blame: Apparently the export thing did not finish and nobody told us, so now we have a pile of half-done files. I will check the logs by 2 pm and send a list of affected files. We should not retry anything until that list is ready because there might already be completed exports mixed in.'),
    dict(id='csv_parse_code', at=6.4, budget=320, prompt='In Python, read CSV text with columns name and score, ignore rows with an empty score, reject invalid nonempty scores with the row number, and return the three highest-scoring names with alphabetical tie-breaking. Use the standard library and explain how quoted commas are handled.'),
]

p = argparse.ArgumentParser()
p.add_argument('--base', required=True)
p.add_argument('--out', required=True)
p.add_argument('--label', required=True)
a = p.parse_args()
base = a.base.rstrip('/').removesuffix('/v1')
out = Path(a.out)
out.mkdir(parents=True, exist_ok=False)
def save(name, value):
    (out / name).write_text(json.dumps(value, indent=2) + '\n')
def req(route, body=None):
    request = urllib.request.Request(base + route, data=None if body is None else json.dumps(body).encode(), headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)
def idle():
    for _ in range(100):
        loads = req('/v1/loads')
        if all(x['num_running_reqs'] == x['num_waiting_reqs'] == 0 for x in loads['loads']):
            return loads
        time.sleep(.1)
    raise AssertionError(loads)
save('loads-before.json', idle())
info = req('/server_info')
assert (info['tp_size'], info['ep_size'], info['context_length'], info['max_total_tokens'], info['max_running_requests'], info['speculative_dspark_block_size']) == (4,4,1000000,4000000,8,5)
save('server-info.json', info)
save('protocol.json', dict(cases=CASES, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          method='Twelve unseen heterogeneous requests, fixed staggered arrivals, one excluded warmup cohort and three measured trials. Native streaming reports exact completion and verification counters. Report end-to-end latency and cohort throughput; do not substitute these numbers for the original OpenAI benchmark. Every request and trial is retained. Housekeeping waits occur outside the measured arrival/finish intervals.'))
inputs = []
for case in CASES:
    rendered = req('/v1/tokenize', dict(model='deepseek-v41-flash', messages=[dict(role='user',content=case['prompt'])], chat_template_kwargs={'thinking':False}))
    inputs.append(rendered)
save('inputs.json', inputs)
trials = []
warmups = []
for trial in [0,1,2,3]:
    idle()
    origin = time.perf_counter() + 0.2
    def one(i):
        case = CASES[i]
        time.sleep(max(0, origin + case['at'] - time.perf_counter()))
        body = dict(input_ids=inputs[i]['tokens'], sampling_params=dict(temperature=0,max_new_tokens=case['budget']), stream=True)
        request = urllib.request.Request(base + '/generate', data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
        started = time.perf_counter()
        first = None
        events = 0
        last = None
        with urllib.request.urlopen(request, timeout=240) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    break
                value = json.loads(data)
                assert isinstance(value, dict) and 'error' not in value, value
                events += 1
                if value.get('text') and first is None:
                    first = time.perf_counter()
                last = value
        ended = time.perf_counter()
        assert last and first is not None and last['text'].strip(), (case['id'], last)
        meta = last['meta_info']
        assert meta['completion_tokens'] > 0 and meta.get('num_retractions',0) == 0
        checked = None
        check_error = None
        if 'expected' in case:
            try:
                actual = json.loads(last['text'].strip())
                checked = actual == case['expected']
            except (ValueError, TypeError) as error:
                checked = False
                check_error = repr(error)
        steps = meta.get('spec_verify_ct', 0)
        caps = meta.get('spec_cap_lens_histogram')
        assert steps > 0
        if caps:
            assert sum(caps) == steps
            mean_cap = sum(k*n for k,n in enumerate(caps))/steps
        else:
            mean_cap = 6.0
        row = dict(id=case['id'], arrival_target_s=case['at'], actual_start_s=started-origin, finish_s=ended-origin,
                    ttft_s=first-started, total_s=ended-started, stream_events=events, prompt_tokens=inputs[i]['count'],
                    verified_rows_per_step=mean_cap, completion_tokens_per_verify=meta['completion_tokens']/steps,
                    exact_check_passed=checked, exact_check_error=check_error, response=last)
        save(f'trial{trial}-{case["id"]}.json', row)
        return row
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(CASES)) as pool:
        responses = list(pool.map(one, range(len(CASES))))
    idle()
    wall = max(r['finish_s'] for r in responses) - min(r['actual_start_s'] for r in responses)
    tokens = sum(r['response']['meta_info']['completion_tokens'] for r in responses)
    row = dict(trial=trial, responses=responses, wall_s=wall, completion_tokens=tokens, cohort_tok_s=tokens/wall,
               mean_latency_s=statistics.mean(r['total_s'] for r in responses), median_latency_s=statistics.median(r['total_s'] for r in responses),
               exact_checks_passed=all(r['exact_check_passed'] is not False for r in responses))
    (warmups if trial == 0 else trials).append(row)
    save('results.json', dict(label=a.label, excluded_warmups=warmups, trials=trials))
    print(json.dumps({k:v for k,v in row.items() if k!='responses'}), flush=True)
save('loads-after.json', idle())
save('completion.json', dict(status='PASS', exact_checks_passed=all(t['exact_checks_passed'] for t in trials),
                            finished=datetime.datetime.now(datetime.timezone.utc).isoformat()))
