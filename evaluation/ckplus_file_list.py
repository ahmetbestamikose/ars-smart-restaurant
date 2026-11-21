"""Build the file list of the local CK+48 folder (class, subject, sequence, frame) and check it.

The list is generated locally from your licensed copy; it is not distributed with the repository.
Usage:  python evaluation/ckplus_file_list.py --data data/ckplus48 --out data/ckplus48_file_list.csv
"""
import argparse, collections, csv, glob, os

EXPECTED = {'anger': 135, 'contempt': 54, 'disgust': 177, 'fear': 75, 'happy': 207, 'sadness': 84, 'surprise': 249}


def build(root):
    rows = []
    for c in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
        for f in sorted(glob.glob(os.path.join(root, c, '*.png'))):   # same order as fer_protocols.py
            name = os.path.basename(f)
            s, q, fr = name[:-4].split('_')
            rows.append(dict(path=f'{c}/{name}', **{'class': c}, subject=s, sequence=f'{s}_{q}', frame=int(fr)))
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--data', default='data/ckplus48')
    ap.add_argument('--out', default='data/ckplus48_file_list.csv'); args = ap.parse_args()
    rows = build(args.data)
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['path', 'class', 'subject', 'sequence', 'frame']); w.writeheader(); w.writerows(rows)
    counts = collections.Counter(r['class'] for r in rows)
    print('images', len(rows), 'subjects', len({r['subject'] for r in rows}), 'sequences', len({r['sequence'] for r in rows}))
    print('per class', dict(counts))
    print('matches the article (981 images, 118 subjects, 327 sequences):',
          dict(counts) == EXPECTED and len({r['subject'] for r in rows}) == 118 and len({r['sequence'] for r in rows}) == 327)
