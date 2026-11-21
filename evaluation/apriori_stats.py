"""Uncertainty and paired comparison for the temporal hold-out of apriori_eval.py.

For the same 1 099 March baskets (same hidden items, seed 123) it reports, for HR@1/3/5:
95% bootstrap confidence intervals (2 000 resamples of baskets) for rules and popularity, the
bootstrap interval of their difference, and an exact McNemar test on the discordant baskets.
It also counts baskets that received fewer than five rule suggestions (lists are not padded).

Usage:  python evaluation/apriori_stats.py --data data/maven_restaurant_orders --out evaluation/results
"""
import argparse, collections, json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import binomtest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apriori_lib import freq, rules

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='data/maven_restaurant_orders'); ap.add_argument('--out', default='evaluation/results')
ap.add_argument('--seed', type=int, default=123); ap.add_argument('--boot', type=int, default=2000)
args = ap.parse_args()
d = pd.read_csv(os.path.join(args.data, 'order_details.csv'), encoding='utf-8-sig').dropna(subset=['item_id'])
d['dt'] = pd.to_datetime(d.order_date, format='%m/%d/%y')
o = d.groupby('order_id').agg(items=('item_id', lambda s: frozenset(s.astype(int))), dt=('dt', 'first'))
tr, te = o[o.dt < '2023-03-01'], o[o.dt >= '2023-03-01']
Rt = rules(freq(list(tr['items']), 0.002), 0.1)
pop = collections.Counter(i for t in tr['items'] for i in t)
rng = np.random.default_rng(args.seed)
R, P, nsug = [], [], []
for t in te['items']:
    if len(t) < 2: continue
    t = list(t); h = t[rng.integers(len(t))]; ctx = frozenset(x for x in t if x != h)   # same draw as apriori_eval.py
    sc = {}
    for A, B, s, c, l in Rt:
        if A <= ctx:
            for b in B:
                if b not in ctx: sc[b] = max(sc.get(b, 0), c)
    r = [i for i, _ in sorted(sc.items(), key=lambda x: -x[1])][:5]
    p = [i for i, _ in pop.most_common() if i not in ctx][:5]
    nsug.append(len(r)); R.append([h in r[:k] for k in (1, 3, 5)]); P.append([h in p[:k] for k in (1, 3, 5)])
R, P, nsug = np.array(R), np.array(P), np.array(nsug); n = len(R)
idx = np.random.default_rng(1).integers(0, n, (args.boot, n))
ci = lambda v: [float(x) for x in np.percentile(v[idx].mean(1) * 100, [2.5, 97.5])]
out = dict(n_baskets=n, baskets_with_fewer_than_5_rule_suggestions=int((nsug < 5).sum()),
           rule_list_length_counts={int(k): int(v) for k, v in enumerate(np.bincount(nsug)) if v})
for j, k in enumerate((1, 3, 5)):
    r, p = R[:, j], P[:, j]; b, c = int((r & ~p).sum()), int((~r & p).sum())
    out[f'HR@{k}'] = dict(rules=float(r.mean() * 100), rules_ci=ci(r), popularity=float(p.mean() * 100), popularity_ci=ci(p),
                          diff_pop_minus_rules_ci=[float(x) for x in np.percentile((p[idx].mean(1) - r[idx].mean(1)) * 100, [2.5, 97.5])],
                          discordant_rules_only=b, discordant_popularity_only=c, mcnemar_exact_p=float(binomtest(b, b + c).pvalue))
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(args.out, 'apriori_stats.json'), 'w'), indent=1)
