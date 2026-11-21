"""Offline evaluation of the Apriori item suggestions on the Maven Analytics restaurant orders.

Usage:
  python evaluation/apriori_eval.py --data data/maven_restaurant_orders --out evaluation/results

1. Rule statistics on all orders (minimum support 0.002, minimum confidence 0.1, itemsets <= 3).
2. Temporal hold-out: rules mined on January-February 2023, tested on March 2023. From every March
   order with at least two distinct items one item is hidden at random and five suggestions are
   requested for the rest. Methods: rules (ranked by the highest confidence of a matching rule),
   popularity (most frequent Jan-Feb items not in the basket), and random suggestions (expected
   hit rate computed exactly as the mean of min(k, C)/C over baskets, C = number of candidate items).
"""
import argparse, collections, json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apriori_lib import freq, rules

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='data/maven_restaurant_orders')
ap.add_argument('--out', default='evaluation/results')
ap.add_argument('--seed', type=int, default=123)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

d = pd.read_csv(os.path.join(args.data, 'order_details.csv'), encoding='utf-8-sig')
n_lines, n_null = len(d), int(d.item_id.isna().sum())
d = d.dropna(subset=['item_id'])
m = pd.read_csv(os.path.join(args.data, 'menu_items.csv'), encoding='utf-8-sig')
name = dict(zip(m.menu_item_id, m.item_name)); n_items = len(m)
d['dt'] = pd.to_datetime(d.order_date, format='%m/%d/%y')
orders = d.groupby('order_id').agg(items=('item_id', lambda s: frozenset(s.astype(int))), dt=('dt', 'first'))
out = dict(n_order_lines=n_lines, n_lines_without_item=n_null, n_menu_items=n_items)

F = freq(list(orders['items']), 0.002); R = rules(F, 0.1)
out['full'] = dict(n_tx=len(orders), n_itemsets=len(F), n_rules=len(R),
                   size_counts=dict(sorted(collections.Counter(len(s) for s in F).items())),
                   conf_quartiles=list(np.percentile([r[3] for r in R], [0, 25, 50, 75, 100])),
                   lift_quartiles=list(np.percentile([r[4] for r in R], [0, 25, 50, 75, 100])),
                   frac_lift_gt1=float(np.mean([r[4] > 1 for r in R])))
fmt = lambda r: dict(A=[name[i] for i in sorted(r[0])], B=[name[i] for i in sorted(r[1])], sup=r[2], conf=r[3], lift=r[4])
out['top_lift'] = [fmt(r) for r in sorted(R, key=lambda r: -r[4])[:8]]

tr = orders[orders.dt < '2023-03-01']; te = orders[orders.dt >= '2023-03-01']
Rt = rules(freq(list(tr['items']), 0.002), 0.1)
pop = collections.Counter(i for t in tr['items'] for i in t)
all_items = set(m.menu_item_id.astype(int))
rng = np.random.default_rng(args.seed)

def rec_rules(ctx, k):
    sc = {}
    for A, B, s, c, l in Rt:
        if A <= ctx:
            for b in B:
                if b not in ctx: sc[b] = max(sc.get(b, 0), c)
    return [i for i, _ in sorted(sc.items(), key=lambda x: -x[1])][:k]

def rec_pop(ctx, k): return [i for i, _ in pop.most_common() if i not in ctx][:k]

K = (1, 3, 5); hit = {k: dict(rules=0, popularity=0, random_expected=0.0) for k in K}; n = 0
for t in te['items']:
    if len(t) < 2: continue
    # iterate the frozenset in CPython order (deterministic for small ints), as in the article run
    t = list(t); h = t[rng.integers(len(t))]; ctx = frozenset(x for x in t if x != h)
    n += 1
    r5, p5 = rec_rules(ctx, 5), rec_pop(ctx, 5); C = len(all_items - ctx)
    for k in K:
        hit[k]['rules'] += h in r5[:k]; hit[k]['popularity'] += h in p5[:k]; hit[k]['random_expected'] += min(k, C) / C
out['holdout'] = dict(n_train_orders=len(tr), n_test_orders=len(te), n_eval_baskets=n, n_rules_train=len(Rt),
                      hit_rate={k: {a: v / n for a, v in r.items()} for k, r in hit.items()})
print(json.dumps(out, indent=1, default=str))
json.dump(out, open(os.path.join(args.out, 'apriori_eval.json'), 'w'), indent=1, default=str)
