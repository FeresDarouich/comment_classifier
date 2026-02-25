# CPU-friendly, saves tokenizer + best model

import os
import json
import random
import argparse
from dataclasses import asdict
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from Tockenizer import WordTokenizer
from Model import build_model, ModelConfig


#___________________Utils________________________

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_csv_text_label(path: str) -> List[Tuple[str,int]]:
    """
    csv reader. Expects header text,label
    """
    import csv
    rows: List[Tuple[str,int]] = []
    with open(path, "r", encoding ="utf-8") as f:
        reader = csv.DictReader(f)
        if "text" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise Exception(f"csv must contain columns text, label found: {reader.fieldnames}")
        for r in reader:
            text = (r["text"] or "").strip()
            label = int(r["label"])
            rows.append((text,label))
    return rows

def train_val_split(rows: List[Tuple[str,int]], val_ratio: float, seed: int) ->Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    n_val = int(len(rows) * val_ratio)
    val = rows[:n_val]
    train = rows [n_val:]
    return train, val

def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    # Inverse frequency weights
    counts = [0] * num_classes
    for y in labels:
        counts[y] += 1
    total = sum(counts)
    # avoid div by zero
    weights = []
    for c in counts:
        if c == 0:
            weights.append(1.0)
        else: 
            weights.append(total /(num_classes * c))
    return torch.tensor(weights,dtype=torch.float32)

def f1_macro(preds: List[int], gold: List[int], num_classes: int) -> float:
    # simple macro-F1 without sklearn
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    for p, y in zip(preds, gold):
        if p == y:
            tp[y] +=1
        else:
             fp[p] +=1
             fn[y] += 1
    f1s = []
    for c in range(num_classes):
        precision = tp[c] /(tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        recall = tp[c] /(tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / num_classes if num_classes > 0 else 0.0

#_________________Dataset_____________

class commentDataset(Dataset):
    def __init__(self, rows: List[Tuple[str, int]], tokenizer: WordTokenizer):
        self.rows = rows
        self.tok = tokenizer
    
    def __len__(self) -> int:
        return len(self.rows)
    
    def __getitem__(self, idx: int):
        text, label = self.rows[idx]
        input_ids = self.tok.encode(text)
        attn = self.tok.attention_mask(input_ids)
        return (
            torch.tensor(input_ids, dtype = torch.long),
            torch.tensor(attn, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )
    
#_______________Train/Eval_______________

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> dict:
    model.eval()
    total_loss = 0.0
    n = 0
    preds_all: List[int] = []
    gold_all: List[int] = []
    ce = nn.CrossEntropyLoss()

    for input_ids, attn, labels in loader:
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        labels = labels.to(device)

        logits = model(input_ids, attn)
        loss = ce(logits, labels)

        bs = labels.size(0)
        total_loss  += loss.item() * bs 
        n += bs

        preds = torch.argmax(logits, dim = 1)
        preds_all.extend(preds.cpu().tolist())
        gold_all.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(1, n)
    acc = sum(int(p == y) for p, y in zip(preds_all, gold_all)) / max (1, len(gold_all))
    f1 = f1_macro(preds_all, gold_all, num_classes=num_classes)
    return {"loss": avg_loss, "acc": acc, "f1_macro": f1}


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cpu")

    rows = read_csv_text_label(args.data_csv)
    if len(rows) < 10:
        raise ValueError("Dataset too small; need at least ~10 rows.")
    
    train_rows, val_rows = train_val_split(rows, args.val_ratio, args.seed)

    # build tokenizer vocab on Train only
    tok = WordTokenizer(vocab_size=args.vocab_size, max_len=args.max_len, add_cls=True, lowercase = True)
    tok.build_vocab([t for t,_ in train_rows])

    num_classes = args.num_classes
    model = build_model(
        vocab_size=len(tok.word2id),
        max_len=args.max_len,
        num_classes=num_classes,
        pad_id=tok.word2id[tok.PAD],
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        use_cls_pooling=True,
    ).to(device)

    train_ds = commentDataset(train_rows, tok)
    val_ds = commentDataset(val_rows, tok)

    train_loader = DataLoader(train_ds, batch_size = args.batch_size, shuffle=True, num_workers = 0)
    val_loader = DataLoader(val_ds,batch_size=args.batch_size, shuffle=False,num_workers=0)
    
    # Loss (optionally weighted)
    if args.class_weighting:
        weights = compute_class_weights([y for _,y in train_rows], num_classes = num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)

    best_f1 = -1.0
    os.makedirs(args.out_dir, exist_ok=True)
    # save tokenizer now (so predictor can load same vocab)
    tok_path = os.path.join(args.out_dir, "tokenizer.json")
    tok.save(tok_path)

    # save config
    cfg_path = os.path.join(args.out_dir, "train_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    for epoch in  range(1, args.epochs +1):
        model.train()
        total_loss = 0.0
        n = 0

        for input_ids, attn, labels in train_loader:
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attn)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            n += bs
        
        train_loss = total_loss/max(1,n)
        val_metrics = evaluate(model, val_loader, device, num_classes=num_classes)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['acc']:.4f} | "
            f"val_f1={val_metrics['f1_macro']:.4f}"
        )

        # Save best by F1
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            ckpt = {
                "model_state_dict": model.state_dict(),
                "vocab_size": len(tok.word2id),
                "max_len": args.max_len,
                "num_classes": num_classes,
                "pad_id": tok.word2id[tok.PAD],
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "d_ff": args.d_ff,
                "dropout": args.dropout,
                "use_cls_pooling": True,
            }
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))

    print(f"Done. Best val F1: {best_f1:.4f}")
    print(f"Saved: {args.out_dir}/best.pt and tokenizer.json")

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data_csv", type=str, required=True,help="CSV with columns text,label")
    p.add_argument("--out_dir", type=str,default="runs/comment_cls")
    # Tokenizer
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--max_len", type=int, default=64)

    # Model
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--d_ff", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)

    # Train
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--class_weighting", action="store_true")
    return p

if __name__== "__main__":
    args = build_argparser().parse_args()
    train(args)