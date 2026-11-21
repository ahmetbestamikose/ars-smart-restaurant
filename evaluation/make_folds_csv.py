"""Write the Protocol B fold assignment (test fold and validation subjects per fold) to CSV.

Uses the same scikit-learn splitters and seeds as fer_protocols.py, but only needs the file list.
Generated locally (not distributed). First run evaluation/ckplus_file_list.py on your licensed copy.
Usage:  python evaluation/make_folds_csv.py --files data/ckplus48_file_list.csv --out data/protocolB_folds.csv
"""
import argparse, csv
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
ap = argparse.ArgumentParser(); ap.add_argument('--files', default='data/ckplus48_file_list.csv')
ap.add_argument('--out', default='data/protocolB_folds.csv'); ap.add_argument('--seed', type=int, default=123)
args = ap.parse_args()
rows = list(csv.DictReader(open(args.files)))
classes = sorted({r['class'] for r in rows})
y = np.array([classes.index(r['class']) for r in rows]); subj = np.array([r['subject'] for r in rows])
X = np.zeros((len(rows), 1))
role = {}
sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
for k, (tr, te) in enumerate(sgk.split(X, y, subj)):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed + k)
    _, va_i = next(gss.split(X[tr], y[tr], subj[tr]))
    for i in te: role.setdefault(i, {})['test_fold'] = k + 1
    for i in tr[va_i]: role.setdefault(i, {}).setdefault('val_folds', []).append(k + 1)
with open(args.out, 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['path', 'class', 'subject', 'test_fold', 'validation_in_folds'])
    for i, r in enumerate(rows):
        w.writerow([r['path'], r['class'], r['subject'], role[i]['test_fold'], ';'.join(map(str, role[i].get('val_folds', [])))])
print('wrote', args.out)
