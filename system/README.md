# Prototype source code

The integrated prototype described in Section 3 of the article consists of:

| Folder (planned) | Component | Technology |
|---|---|---|
| `desktop/` | Management dashboard (orders, tables, emotion tab, chef's suggestion, refresh hub) | Java Swing, FlatLaf, JDBC, Maven |
| `emotion_server/` | Camera capture, Haar-cascade face detection, CNN classification, TCP stream (port 9999) | Python, OpenCV, TensorFlow/Keras |
| `backend/` | REST API (`/api`), Apriori engine, emotion and order watchers, Expo push | Python, FastAPI, SQLite |
| `mobile/` | Waiter application (PIN login, tables, orders with suggestions, alerts) | React Native, Expo, TypeScript |

> **Status:** the source code of these components will be added here before the article is
> published. Until then, please contact the corresponding author.

Running notes (to be completed with the code):
- All components share one SQLite file (`restoran.db`, WAL mode); the back end creates missing
  tables at start-up. No database file is distributed.
- The emotion server needs a trained model file (`*.h5`), which is not distributed because it is
  derived from CK+. Train one with `training/` or `evaluation/fer_protocols.py`.
- The waiter PIN and the API address are configured through environment variables; the default PIN
  of the prototype is for demonstration only.
