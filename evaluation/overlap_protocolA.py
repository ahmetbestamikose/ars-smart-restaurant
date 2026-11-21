"""How much of a Protocol A validation split is already 'seen' in training?

Reproduces the thesis pipeline at the file level (no images needed): every class is topped up to
600 images with augmented copies of randomly chosen originals, the pool is shuffled and split
80/20. For each validation image we check whether the training part holds an image of the same
subject, of the same posed sequence, or of the same source frame (the original or an augmented
copy of it). Reported over 20 random splits.

Needs the file list generated locally by evaluation/ckplus_file_list.py (not distributed).
Usage:  python evaluation/overlap_protocolA.py --files data/ckplus48_file_list.csv
"""
import argparse, csv, json, random, collections
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--files', default='data/ckplus48_file_list.csv')
ap.add_argument('--splits', type=int, default=20); ap.add_argument('--target', type=int, default=600)
args = ap.parse_args()
by_class = collections.defaultdict(list)
for r in csv.DictReader(open(args.files)): by_class[r['class']].append(r['path'].split('/')[-1])
res = []
for seed in range(args.splits):
    rng = random.Random(seed); items = []
    for c, fs in sorted(by_class.items()):
        items += fs + [rng.choice(fs) for _ in range(args.target - len(fs))]
    rng.shuffle(items); nv = int(round(0.2 * len(items))); val, tr = items[:nv], items[nv:]
    subj = lambda f: f.split('_')[0]; seq = lambda f: '_'.join(f.split('_')[:2])
    S, Q, Fr = {subj(f) for f in tr}, {seq(f) for f in tr}, set(tr)
    res.append([np.mean([subj(f) in S for f in val]), np.mean([seq(f) in Q for f in val]), np.mean([f in Fr for f in val])])
r = np.array(res)
out = dict(splits=args.splits, same_subject_min=r[:, 0].min(), same_sequence_min=r[:, 1].min(),
           same_sequence_mean=r[:, 1].mean(), same_frame_mean=r[:, 2].mean())
print(json.dumps(out, indent=1))
