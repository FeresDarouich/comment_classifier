import re
import json
from collections import Counter
from typing import List, Dict, Optional


class WordTockenizer:
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
    
    def tockenize(self, text: str) -> List[str]:
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
        tockens = self.tockenize(text)
        if self.add_cls:
            tockens = [self.CLS] + tockens
        ids = [self.word2id.get(tok, self.word2id[self.UNK]) for tok in tockens]
        if len(ids) < self.max_len:
            ids = ids + [self.word2id[self.PAD]] * (self.max_len - len(ids))
        else:
            ids = ids[: self.max_len]
        return ids