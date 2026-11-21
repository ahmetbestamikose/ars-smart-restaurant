# Original training notebook (thesis run, Protocol A)

`thesis_training_colab.ipynb` is the notebook that produced the model used in the prototype
(Google Colab). It is kept unchanged, including its outputs (training curves, confusion matrix and
classification report of the 825 validation images; best validation accuracy 0.989).

Note on evaluation: the notebook oversamples every class to 600 images *before* the 80/20 split and
reports on the same validation split that is used for checkpoint selection. As discussed in the
article, this makes the 98.9% an optimistic, image-level figure. The subject-independent
re-evaluation is in `../evaluation/fer_protocols.py --protocol B`.

To run it outside Colab, replace the Google Drive paths at the top with `data/ckplus48`.
