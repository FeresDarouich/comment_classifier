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

## Project Structure


Project Structure
.
├── tokenizer.py      # Custom word-level tokenizer
├── model.py          # Tiny Transformer classifier
├── train.py          # Training pipeline
├── predictor.py      # Inference script
├── data/
│   └── comments.csv  # Training dataset
├── runs/
│   └── v1/
│       ├── best.pt
│       ├── tokenizer.json
│       └── train_config.json
└── README.md
Requirements

Python 3.9+

PyTorch

Install dependencies:

pip install torch

No additional libraries (pandas, sklearn, etc.) are required.

Dataset Format

Training data must be a CSV file with the following columns:

text,label
"I love this product",0
"This is terrible",1

text → comment string

label → integer class ID (0 to num_classes - 1)

Training

Example training command:

python train.py \
  --data_csv data/comments.csv \
  --out_dir runs/v1 \
  --num_classes 2 \
  --class_weighting
Key Arguments
Argument	Description
--vocab_size	Maximum vocabulary size (default: 10000)
--max_len	Maximum tokens per comment (default: 64)
--d_model	Embedding dimension (default: 128)
--n_heads	Number of attention heads (default: 4)
--n_layers	Number of Transformer blocks (default: 2)
--d_ff	Feed-forward hidden size (default: 256)
--dropout	Dropout rate (default: 0.1)
--epochs	Number of training epochs
--batch_size	Batch size
--class_weighting	Enable class balancing
Saved Artifacts

After training, the following files are generated:

best.pt → best model checkpoint (selected by validation macro-F1)

tokenizer.json → saved vocabulary

train_config.json → training configuration

These files are required for inference.

Inference

Example:

python predictor.py \
  --model runs/v1/best.pt \
  --tokenizer runs/v1/tokenizer.json \
  --text "This comment is awful" \
  --labels '["non_toxic","toxic"]'

Example output:

{
  "label_id": 1,
  "label_name": "toxic",
  "confidence": 0.87,
  "probs": [0.13, 0.87]
}
Model Architecture

The model consists of:

Word Embedding Layer

Learned Positional Embeddings

N Transformer Encoder Blocks

CLS Pooling

Linear Classification Layer

Input Shapes
input_ids      : [B, T]
attention_mask : [B, T]
Output Shape
logits : [B, num_classes]
Recommended CPU Configuration

A balanced configuration:

vocab_size = 10000
max_len    = 64
d_model    = 128
n_heads    = 4
n_layers   = 2
d_ff       = 256
dropout    = 0.1

This provides good performance while remaining CPU-friendly.

Training Tips

Build vocabulary on training data only.

Use macro-F1 for evaluation if classes are imbalanced.

Overfit a small batch first to verify training works.

Ensure tokenizer and model max_len are consistent.
