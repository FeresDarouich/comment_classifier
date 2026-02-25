import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Union

import torch
import torch.nn.functional as F

from Tockenizer import WordTokenizer
from Model import build_model

@dataclass
class Prediction:
    label_id: int
    confidence: float
    probs: List[float]
    label_name: Optional[str] = None

class CommentPredictor:
    def __init__(
            self,
            model_path: str,
            tokenizer_path: str,
            label_names: Optional[List[str]] = None,
            device: str = "cpu",
    ):
        self.device = torch.device(device)
        # load tokenizer
        self.tok = WordTokenizer.load(tokenizer_path)
        # load checkpoints
        ckpt = torch.load(model_path, map_location=self.device)
        # rebuild model
        self.model = build_model(
            vocab_size = ckpt["vocab_size"],
            max_len = ckpt["max_len"],
            num_classes = ckpt["num_classes"],
            pad_id=ckpt["pad_id"],
            d_model=ckpt["d_model"],
            n_heads=ckpt["n_heads"],
            n_layers=ckpt["n_layers"],
            d_ff=ckpt["d_ff"],
            dropout=ckpt["dropout"],
            use_cls_pooling=ckpt.get("use_cls_pooling", True),
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # Optional human-readable labels
        if self.label_names is not None and len(self.label_names) != ckpt["num_classes"]:
            raise ValueError(
                f"label_names length ({len(self.label_names)}) must equal num classes ({ckpt["num_classes"]})"
            )
        # Safety: ensure tokenizer max_len matches model max_len
        if self.tok.max_len != ckpt["max_len"]:
            raise ValueError(
                f"Tokenizer max_len ({self.tok.max_len}) != checkpoint max_len ({ckpt['max_len']}). "
                "Fix by training and saving consistent max_len."
            )
    @torch.no_grad()
    def predict_one(self, text: str) -> Prediction:
        ids= self.tok.encode(text)
        mask = self.tok.attention_mask(ids)

        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attn = torch.tensor([mask], dtype = torch.long, device = self.device)

        logits = self.model(input_ids, attn)
        probs = F.softmax(logits, dim = -1)[0].cpu().tolist()

        label_id = int(max(range(len(probs)), key=lambda i: probs[i]))
        confidence = float(probs[label_id])

        label_name = None
        if self.label_names is not None:
            label_name = self.label_names[label_id]

        return Prediction(
            label_id=label_id,
            confidence=confidence,
            probs =[float(p) for p in probs],
            label_name = label_name,
        )
    @torch.no_grad()
    def predict_batch(self, texts: List[str],batch_size: int = 64) -> List[Prediction]:
        # simple CPU batching (no DataLoader needed)
        out: List[Prediction] = []
        C = None
        for i in range(0, len(texts), batch_size):
            chunk = texts[i: i +batch_size]

            ids_batch = [self.tok.encode(t) for t in chunk]
            mask_batch = [self.tok.attention_mask(ids) for ids in ids_batch]

            input_ids = torch.tensor(ids_batch, dtype=torch.long,device=self.device)
            attn = torch.tensor(mask_batch, dtype=torch.long, device=self.device)

            logits = self.model(input_ids, attn)
            probs = F.softmax(logits, dim=-1).cpu()
            if C is None:
                C = probs.size(1)

            for row in probs.tolist():
                label_id = int(max(range(len(row)), key=lambda k:row[k]))
                confidence = float(row[label_id])
                label_name = self.label_names[label_id] if self.label_names else None
                out.append(
                    Prediction(
                        label_id=label_id,
                        confidence = confidence,
                        probs = [float(p) for p in row],
                        label_name= label_name,
                    )
                )
        return out

def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model", type = str, required = True, help = "Path to best.pt")
    p.add_argument("--tokenizer", type=str, required=True, help = "Path to tokenizer.json")
    p.add_argument("--text", type=str, default=None,help="Single text to classify")
    p.add_argument("--labels", type=str, default=None, help="Json List e.g. \'['ok','toxic']\'")
    args = p.parse_args()

    label_names = None
    if args.labels:
        label_names = json.loads(args.labels)
    pred = CommentPredictor(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        label_names=label_names,
        device = "cpu"
    )
    if args.text is None:
        raise ValueError("Provide text please ! ")
    y= pred.predict_one(args.text)
    print(
        {
            "label_id": y.label_id,
            "label_name": y.label_name,
            "confidence": y.confidence,
            "probs": y.probs,
        }
    )
if __name__=="__main__":
    main()