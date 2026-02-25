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
Overfit a small batch first to verify training works.

Ensure tokenizer and model max_len are consistent.
