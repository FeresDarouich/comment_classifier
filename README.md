# Tiny Transformer Comment Classifier (CPU-Friendly)

A lightweight Transformer-based text classification system built from scratch using PyTorch. Designed to be simple, efficient, and runnable entirely on CPU without pretrained dependencies.

---

## Overview

This project implements:

- A custom word-level tokenizer  
- A Tiny Transformer encoder classifier  
- A full training pipeline  
- A prediction (inference) script  
- Checkpoint + tokenizer saving for reproducibility  

The classifier takes raw text comments and predicts their class (e.g., toxic vs non-toxic, positive vs negative, spam vs not spam).

---

## Architecture

The model consists of:

1. Word Embedding Layer  
2. Learned Positional Embeddings  
3. Stacked Transformer Encoder Blocks  
4. CLS Token Pooling  
5. Linear Classification Head  

### Input

- `input_ids` → `[B, T]`
- `attention_mask` → `[B, T]`

Where:
- `B` = batch size  
- `T` = max sequence length  

### Output

- `logits` → `[B, num_classes]`

---

## Setup

### Option A (recommended): Poetry

Install dependencies using Poetry:

```bash
poetry install --no-root
```

### Option B: pip + venv

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

---

## Makefile shortcuts

If you have `make` available (Git Bash / WSL on Windows, or any Unix-like shell), you can use:

```bash
make install
make train
make predict TEXT="thanks for the help" LABELS=ok,toxic
```

Defaults:

- `DATA=data/sample_comments_ok_toxic.csv`
- `OUT=runs/comment_cls`
- `NUM_CLASSES=2`

---

## Sample data

A small sample dataset is included at:

- `data/sample_comments_ok_toxic.csv`

It is a UTF-8 CSV with this header:

- `text,label`

Label mapping used in that file:

- `0` → non-toxic / ok
- `1` → toxic / abusive

---

## Train

Train the model from a CSV file:

```bash
python src/Trainer.py --data_csv data/sample_comments_ok_toxic.csv --num_classes 2
```

Outputs are saved to `runs/comment_cls/` by default:

- `best.pt` (best checkpoint by macro-F1)
- `tokenizer.json` (vocab + tokenizer settings)
- `train_config.json` (the CLI arguments used)

Common useful flags:

- `--epochs 8` (default)
- `--max_len 64` (default)
- `--class_weighting` (enables inverse-frequency weighting)

---

## Predict

After training, run inference on a single comment:

```bash
python src/Predict.py --model runs/comment_cls/best.pt --tokenizer runs/comment_cls/tokenizer.json --text "this is amazing, thanks!" --labels ok,toxic
```

If you omit `--labels`, the output includes `label_id` only. `--labels` also accepts a JSON list (e.g. `["ok","toxic"]`).

---

## Your own dataset format

`src/Trainer.py` expects a UTF-8 CSV with:

- Header columns: `text,label`
- `text`: the raw comment string
- `label`: an integer in `[0, num_classes-1]`

Note: the trainer currently requires at least ~10 rows.

