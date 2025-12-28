import re
import json
from collections import Counter
from typing import List, Dict, Optional


class WordTokenizer:
    def __init__(
            self,
            vocab_size: int = 10_000,
            max_len: int = 64,
            lowercase: bool = True,
            add_cls: bool = True,
    ):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.lowercase = lowercase
        self.add_cls = add_cls

        # special tockens
        self.PAD = "<pad>"
        self.UNK = "<unk>"
        self.CLS = "<cls>"

        self.word2id: Dict[str, int] = {}
        self.id2word: Dict[int, str] = {}

        self._pattern = re.compile(r"[A-Za-z0-9']+")
    
    def tokenize(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()
        return self._pattern.findall(text)
    
    def build_vocab(self, texts: List[str]) -> None:
        counter = Counter()
        for t in texts:
            counter.update(self.tokenize(t))
        special= [self.PAD,self.UNK] + ([self.CLS] if self.add_cls else [])
        remaining = max(0, self.vocab_size - len(special))

        most_common = [ w for w, _ in counter.most_common(remaining)]
        vocab = special + most_common

        self.word2id = {w: i for i,w in enumerate(vocab)}
        self.id2word = {i: w for w,i in self.word2id.items()}

    def encode(self, text: str) -> List[int]:
        if not self.word2id:
            raise ValueError(" Vocab not build.Call build_vocab(texts) first.")
        tockens = self.tokenize(text)
        if self.add_cls:
            tockens = [self.CLS] + tockens
        ids = [self.word2id.get(tok, self.word2id[self.UNK]) for tok in tockens]
        if len(ids) < self.max_len:
            ids = ids + [self.word2id[self.PAD]] * (self.max_len - len(ids))
        else:
            ids = ids[: self.max_len]
        return ids
    
    def attention_mask(self, ids: List[int]):
        """1 for real tockens, 0 for padding."""
        if not self.word2id:
            raise ValueError("Vovab not built.")
        pad_id = self.word2id[self.PAD]
        return [0 if x == pad_id else 1 for x in ids]
    
    def decode(self, ids: List[int],skip_special: bool = True) -> str:
        if not self.id2word:
            raise ValueError("Vocab not built.")
        words = []
        for i in ids:
            w = self.id2word.get(int(i), self.UNK)
            if  skip_special and w in {self.PAD,  self.UNK, self.CLS}:
                continue
            words.append(w)
        return " ".join(words)
    
    def save(self, path: str) -> None:
        payload = {
            "vocab_size": self.vocab_size,
            "max_len": self.max_len,
            "lowercase": self.lowercase,
            "add_cls": self.add_cls,
            "word2id": self.word2id,
        }
        with open(path, "w", encoding = "utf-8") as f:
            json.dump(payload, f, ensure_ascii = False, indent = 2)
    @classmethod
    def load(cls, path:str) -> "WordTokenizer":
        with open(path, "r", encoding = "utf-8") as f:
            payload = json.load(f)
        tok = cls(
            vocab_size = payload["vocab_size"],
            max_len = payload["max_len"],
            lowercase = payload["lowercase"],
            add_cls = payload["add_cls"],
        )
        tok.word2id = {k: int(v) for k,v in payload["word2id"].items()}
        tok.id2word = {int(v): k for k,v in tok.word2id.items()}
        return tok
    
if __name__ == "__main__":
    texts = [
        "Hello, World!",
        "Hello there. Don't panic.",
        "Transformers are cool; transformers learn attention.",
    ]

    tok = WordTokenizer(vocab_size=50, max_len=12, add_cls=True)
    tok.build_vocab(texts)

    s = "Hello there, unknown_word!!!"
    ids = tok.encode(s)
    mask = tok.attention_mask(ids)

    print("Text:", s)
    print("Tokens:", tok.tokenize(s))
    print("IDs:", ids)
    print("Mask:", mask)
    print("Decoded:", tok.decode(ids))