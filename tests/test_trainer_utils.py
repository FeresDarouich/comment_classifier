import csv
from pathlib import Path

import torch

from Trainer import read_csv_text_label, compute_class_weights, f1_macro


def test_read_csv_text_label(tmp_path: Path):
    p = tmp_path / "data.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerow(["hi", "0"])
        w.writerow(["bye", "1"])

    rows = read_csv_text_label(str(p))
    assert rows == [("hi", 0), ("bye", 1)]


def test_compute_class_weights_inverse_frequency():
    labels = [0, 0, 0, 1]
    w = compute_class_weights(labels, num_classes=2)
    assert isinstance(w, torch.Tensor)
    assert w.shape == (2,)
    # minority class should have higher weight
    assert float(w[1]) > float(w[0])


def test_f1_macro_simple_case():
    preds = [0, 1, 0, 1]
    gold = [0, 1, 1, 1]
    f1 = f1_macro(preds, gold, num_classes=2)
    assert 0.0 <= f1 <= 1.0
