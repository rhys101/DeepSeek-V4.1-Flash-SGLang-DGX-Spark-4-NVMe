"""Compare native speculation counters without treating them as streaming timings."""
import argparse
import collections
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('before')
p.add_argument('after')
p.add_argument('--out', required=True)
a = p.parse_args()

def collect(path):
    data = json.loads(Path(path).read_text())
    rows = {}
    for case in data['cases']:
        hist = collections.Counter()
        cap = collections.Counter()
        steps = accepted = output = 0
        for request in case['requests']:
            m = request['response']['meta_info']
            n = m['spec_verify_ct']
            h = m['spec_correct_drafts_histogram']
            assert sum(h) == n
            assert sum(i * count for i, count in enumerate(h)) == m['spec_num_correct_drafts']
            hist.update({i: count for i, count in enumerate(h)})
            ch = m.get('spec_cap_lens_histogram')
            if ch:
                assert sum(ch) == n
                cap.update({i: count for i, count in enumerate(ch)})
            else:
                # The baseline is static and verifies gamma plus the bonus row.
                cap[data['gamma'] + 1] += n
            steps += n
            accepted += m['spec_num_correct_drafts']
            output += m['completion_tokens']
        verify_rows = sum(i * count for i, count in cap.items())
        rows[(case['category'], case['c'])] = dict(
            steps=steps, completion_tokens=output, accepted_drafts=accepted,
            completion_tokens_per_verify=output / steps,
            verified_rows_per_step=verify_rows / steps,
            verified_rows_per_output_token=verify_rows / output,
            accepted_drafts_histogram=dict(sorted(hist.items())),
            verify_cap_histogram=dict(sorted(cap.items())),
        )
    return rows

b = collect(a.before)
c = collect(a.after)
assert b.keys() == c.keys()
rows = []
for key, before in b.items():
    after = c[key]
    row = dict(workload=key[0], c=key[1], before=before, after=after,
               verified_rows_per_output_delta_percent=(after['verified_rows_per_output_token'] / before['verified_rows_per_output_token'] - 1) * 100)
    rows.append(row)
    print(f"{key[0]} C{key[1]}: verified rows/step {before['verified_rows_per_step']:.3f} -> {after['verified_rows_per_step']:.3f}; output/step {before['completion_tokens_per_verify']:.3f} -> {after['completion_tokens_per_verify']:.3f}; rows/output {row['verified_rows_per_output_delta_percent']:+.2f}%")
Path(a.out).write_text(json.dumps(dict(
    method='Native per-request counters on chat-tokenized inputs. Static caps are gamma+1; compact caps come from the returned histogram. This measures actual unpadded verification rows, not CUDA graph padding or streaming throughput. Prompt suffixes differ from SparkDash at C8; this is diagnostic evidence.',
    rows=rows), indent=2) + '\n')
