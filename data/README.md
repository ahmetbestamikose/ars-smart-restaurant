# Data

## 1. Facial expression images (CK+, not included)

The expression model was trained and evaluated on 981 grey-scale face crops of 48×48 pixels
from the Extended Cohn–Kanade database (CK+): 118 subjects, 327 posed sequences, the last three
frames of each sequence, seven classes (anger, contempt, disgust, fear, happy, sadness, surprise).

**CK+ images are not redistributed with this repository.** Researchers must obtain access from the
dataset maintainers and comply with the CK+ user agreement, which allows non-commercial research use
only, forbids providing any portion of the database to other parties and requires citation of:

- Kanade, T., Cohn, J.F. and Tian, Y. (2000). Comprehensive database for facial expression analysis. FG 2000, 46–53.
- Lucey, P. et al. (2010). The Extended Cohn-Kanade Dataset (CK+). CVPR Workshops, 94–101.

To reproduce the experiments:

1. Request CK+ through the University of Pittsburgh portal: https://ckplus.jeffcohn.net/
   (according to the user agreement, requests must be made by a faculty member).
2. Build the 48×48 crops from your licensed copy: take the last three frames of each of the 327
   emotion-labelled sequences, convert to grey scale, crop the face with OpenCV's frontal-face Haar
   cascade and resize to 48×48. (The article cites the Kaggle page "CK+48" only to describe this
   48×48 version; this repository does not link to or rely on unofficial copies of CK+.)
3. Place them as

   ```
   data/ckplus48/anger/<subject>_<sequence>_<frame>.png   (original CK+ file names)
   data/ckplus48/contempt/...
   ...
   data/ckplus48/surprise/...
   ```

4. Generate the file list locally and check it against the article (981 images, 118 subjects,
   327 sequences; 135 / 54 / 177 / 75 / 207 / 84 / 249 images per class):

   ```
   python evaluation/ckplus_file_list.py --data data/ckplus48 --out data/ckplus48_file_list.csv
   python evaluation/make_folds_csv.py  --files data/ckplus48_file_list.csv   # Protocol B folds
   ```

   File lists and fold assignments are derived from CK+ and are therefore not shared either.

## 2. Restaurant orders (included)

`maven_restaurant_orders/` – public order data of a fictitious restaurant (Maven Analytics,
GPL-3.0). See the README in that folder.
