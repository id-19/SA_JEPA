# Validate the speaker-probe pool construction alone (no model, no torch encode): proves every
# speaker kept has >=2 windows, so the stratified split cannot raise, and prints the pool size
# and what the blind baseline (1/#speakers) is. Cheap pre-flight for a long run.
import collections
import sys

sys.path.insert(0, '/Users/id19/Programming/ml_ai_and_data_science/SA-JEPA/v2_unfrozen/one-shot')
from train_v0 import split_utts

train_rows, probe_rows = split_utts(seed=0)
by_spk = collections.defaultdict(list)
for r in probe_rows:
    by_spk[r['utt'].split('-')[0]].append(r)
kept = {s: v[:4] for s, v in by_spk.items() if len(v) >= 4}
pool = [r for v in kept.values() for r in v]
yp = [r['utt'].split('-')[0] for r in pool]
counts = collections.Counter(yp)
print(f'probe_rows: {len(probe_rows)} windows / {len(by_spk)} speakers')
print(f'speakers with >=4 held-out windows: {len(kept)} / {len(by_spk)} '
      f'(dropped: {sorted(set(by_spk) - set(kept))})')
print(f'pool: {len(pool)} windows, {len(counts)} speakers, '
      f'windows/speaker min {min(counts.values())} max {max(counts.values())}')
print(f'blind baseline (1/n_speakers): {1/len(counts):.3f}  <- any probe above this is signal')
assert len(counts) >= 2 and min(counts.values()) >= 2, 'stratified split would still raise'
print('OK: stratified split is satisfiable for every seed')
