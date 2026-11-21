"""Re-evaluation of the ARS facial-expression CNN on the 48x48 CK+ crops (CK+48).

Usage:
  python evaluation/fer_protocols.py --data data/ckplus48 --protocol B
  python evaluation/fer_protocols.py --data data/ckplus48 --protocol A

Protocol A  (replication of the thesis notebook): class-wise augmented oversampling to 600
            images per class is done on the pooled data *before* an 80/20 random image-level
            split; the same split is used for checkpoint selection and for reporting.
Protocol B  (subject-independent): 5-fold StratifiedGroupKFold over subject IDs. Oversampling
            and augmentation are applied to training subjects only; early stopping uses a
            validation subset of training subjects; test folds contain only original images of
            unseen subjects.
The CNN architecture and optimiser settings are copied from training/thesis_training_colab.ipynb.
The pre-training augmentation of the notebook (Keras ImageDataGenerator) is re-implemented with
OpenCV affine warps using the same parameters.
"""
import os, sys, json, glob, argparse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--data', required=True, help='folder with one sub-folder of 48x48 PNGs per class (see data/README.md)')
ap.add_argument('--out', default='evaluation/results', help='output folder for JSON results and checkpoints')
ap.add_argument('--protocol', choices=['A', 'B'], required=True)
ap.add_argument('--epochs', type=int, default=60, help='maximum epochs (60 in the paper; use 1-2 for a smoke test)')
ap.add_argument('--seed', type=int, default=123)
ap.add_argument('--warmup', type=int, default=None, help='epochs before early stopping/checkpointing may act (default: 0 for A, 20 for B)')
ap.add_argument('--tag', default=None, help='suffix for the output JSON (e.g. warmup20)')
args = ap.parse_args()
ROOT, OUT, MODE, SEED, EPOCHS = args.data, args.out, args.protocol, args.seed, args.epochs
WARMUP = args.warmup if args.warmup is not None else (0 if MODE == 'A' else min(20, max(EPOCHS - 1, 0)))  # B: small folds need BN warm-up
os.makedirs(OUT, exist_ok=True)

CLASSES = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
X, y, subj = [], [], []
for ci, c in enumerate(CLASSES):
    for f in sorted(glob.glob(os.path.join(ROOT, c, '*.png'))):
        X.append(cv2.imread(f, cv2.IMREAD_GRAYSCALE))
        y.append(ci)
        subj.append(os.path.basename(f).split('_')[0])
X = np.stack(X).astype(np.float32); y = np.array(y); subj = np.array(subj)
print('loaded', X.shape, 'classes', CLASSES, 'subjects', len(set(subj)), flush=True)

rng = np.random.default_rng(SEED)

def augment(img):
    """rotation +-15 deg, zoom 0.15, shift 0.1, horizontal flip, nearest fill (ImageDataGenerator params)."""
    h, w = img.shape
    ang = rng.uniform(-15, 15)
    z = rng.uniform(1 - 0.15, 1 + 0.15)
    tx, ty = rng.uniform(-0.1, 0.1) * w, rng.uniform(-0.1, 0.1) * h
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0 / z)
    M[0, 2] += tx; M[1, 2] += ty
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.5:
        out = out[:, ::-1]
    return out

def oversample(Xs, ys, target):
    Xo, yo = [Xs], [ys]
    for c in range(len(CLASSES)):
        idx = np.where(ys == c)[0]
        need = target - len(idx)
        if need > 0:
            pick = rng.choice(idx, need, replace=True)
            Xo.append(np.stack([augment(Xs[i]) for i in pick])); yo.append(np.full(need, c))
    return np.concatenate(Xo), np.concatenate(yo)

def build():
    tf.keras.utils.set_random_seed(SEED)
    m = models.Sequential([
        layers.Input(shape=(48, 48, 1)),
        layers.RandomFlip('horizontal'), layers.RandomRotation(0.05), layers.Rescaling(1. / 255),
        layers.Conv2D(64, 3, padding='same', activation='relu'), layers.BatchNormalization(), layers.MaxPooling2D(), layers.Dropout(0.2),
        layers.Conv2D(128, 3, padding='same', activation='relu'), layers.BatchNormalization(), layers.MaxPooling2D(), layers.Dropout(0.3),
        layers.Conv2D(256, 3, padding='same', activation='relu'), layers.BatchNormalization(), layers.MaxPooling2D(), layers.Dropout(0.4),
        layers.Flatten(), layers.Dense(512, activation='relu'), layers.BatchNormalization(), layers.Dropout(0.5),
        layers.Dense(len(CLASSES), activation='softmax')])
    m.compile(optimizer=optimizers.Adam(learning_rate=1e-4, clipnorm=0.5), loss='categorical_crossentropy', metrics=['accuracy'])
    return m

class WarmCheckpoint(callbacks.Callback):
    """Save weights with best val_accuracy, considering only epochs after the warm-up."""
    def __init__(self, path, warmup):
        super().__init__(); self.path, self.warmup, self.best = path, warmup, -1.0
    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 > self.warmup and logs['val_accuracy'] > self.best:
            self.best = logs['val_accuracy']; self.model.save_weights(self.path)

def fit_eval(Xtr, ytr, Xva, yva, Xte, yte, tag):
    m = build()
    oh = lambda v: tf.keras.utils.to_categorical(v, len(CLASSES))
    ck = os.path.join(OUT, f'{tag}.weights.h5')
    cbs = [WarmCheckpoint(ck, WARMUP),
           callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True,
                                   start_from_epoch=WARMUP)]
    h = m.fit(Xtr[..., None], oh(ytr), validation_data=(Xva[..., None], oh(yva)), epochs=EPOCHS, batch_size=64,
              shuffle=True, callbacks=cbs, verbose=2)
    m.load_weights(ck)  # notebook reloads the best-val_accuracy checkpoint before reporting
    p = m.predict(Xte[..., None], verbose=0).argmax(1)
    res = dict(tag=tag, n_train=int(len(ytr)), n_val=int(len(yva)), n_test=int(len(yte)),
               acc=float(accuracy_score(yte, p)), macro_f1=float(f1_score(yte, p, average='macro')),
               best_val_acc=float(max(h.history['val_accuracy'])), epochs_run=len(h.history['loss']),
               cm=confusion_matrix(yte, p, labels=range(len(CLASSES))).tolist(),
               report=classification_report(yte, p, labels=range(len(CLASSES)), target_names=CLASSES, output_dict=True, zero_division=0),
               history={k: [float(v) for v in vals] for k, vals in h.history.items()})
    print(tag, 'acc', res['acc'], 'macroF1', res['macro_f1'], flush=True)
    return res, p

if MODE == 'A':
    Xa, ya = oversample(X, y, 600)
    perm = rng.permutation(len(ya)); Xa, ya = Xa[perm], ya[perm]
    nv = int(0.2 * len(ya))
    res, _ = fit_eval(Xa[nv:], ya[nv:], Xa[:nv], ya[:nv], Xa[:nv], ya[:nv], 'A_image_level')
    json.dump(res, open(os.path.join(OUT, f"A{'_' + args.tag if args.tag else ''}.json"), 'w'))
else:
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    allres, y_true_all, y_pred_all = [], [], []
    for k, (tr, te) in enumerate(sgk.split(X, y, subj)):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED + k)
        tr_i, va_i = next(gss.split(X[tr], y[tr], subj[tr]))
        tr2, va = tr[tr_i], tr[va_i]
        assert not (set(subj[tr2]) & set(subj[te])) and not (set(subj[va]) & set(subj[te]))
        target = int(np.bincount(y[tr2]).max())
        Xtr, ytr = oversample(X[tr2], y[tr2], target)
        res, p = fit_eval(Xtr, ytr, X[va], y[va], X[te], y[te], f'B_fold{k+1}')
        res['test_subjects'] = sorted(set(subj[te]))
        allres.append(res); y_true_all += y[te].tolist(); y_pred_all += p.tolist()
        json.dump(allres, open(os.path.join(OUT, 'B.json'), 'w'))
    summ = dict(fold_acc=[r['acc'] for r in allres], fold_f1=[r['macro_f1'] for r in allres],
                pooled_cm=confusion_matrix(y_true_all, y_pred_all).tolist(),
                pooled_report=classification_report(y_true_all, y_pred_all, target_names=CLASSES, output_dict=True, zero_division=0))
    json.dump(summ, open(os.path.join(OUT, 'B_summary.json'), 'w'))
    print('B mean acc', np.mean(summ['fold_acc']), np.std(summ['fold_acc']), 'macroF1', np.mean(summ['fold_f1']), flush=True)
