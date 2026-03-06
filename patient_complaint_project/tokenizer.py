import jieba
from collections import Counter
import os

class Tokenizer:
    """Simple tokenizer/vocab using jieba for Chinese segmentation.

    - build from JSON files containing 'visit_sn' text
    - provides encode(text, max_len) -> list of ids (with [CLS]/[SEP]/[PAD]/[UNK])
    """
    def __init__(self, vocab_path=None, max_vocab=30000, min_freq=2):
        self.special_tokens = ['[PAD]','[UNK]','[CLS]','[SEP]']
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        if vocab_path and os.path.exists(vocab_path):
            self.load_vocab(vocab_path)
        else:
            self.token2id = {t:i for i,t in enumerate(self.special_tokens)}
            self.id2token = {i:t for t,i in self.token2id.items()}

    def build_vocab_from_texts(self, texts):
        counter = Counter()
        for t in texts:
            if not t: continue
            toks = list(jieba.cut(t))
            counter.update(toks)
        # collect tokens with min_freq
        items = [tok for tok, cnt in counter.most_common(self.max_vocab) if cnt >= self.min_freq]
        for tok in items:
            if tok not in self.token2id:
                self.token2id[tok] = len(self.token2id)
        self.id2token = {i:t for t,i in self.token2id.items()}

    def save_vocab(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            for i in range(len(self.id2token)):
                f.write(self.id2token[i] + '\n')

    def load_vocab(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            toks = [l.strip() for l in f if l.strip()]
        self.token2id = {t:i for i,t in enumerate(toks)}
        self.id2token = {i:t for t,i in self.token2id.items()}

    def encode(self, text, max_len=256):
        if not text:
            tokens = ['[CLS]','[SEP]']
        else:
            seg = list(jieba.cut(text))
            seg = seg[:max_len-2]
            tokens = ['[CLS]'] + seg + ['[SEP]']
        ids = [self.token2id.get(t, self.token2id['[UNK]']) for t in tokens]
        if len(ids) < max_len:
            ids = ids + [self.token2id['[PAD]']] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        return ids

    def vocab_size(self):
        return len(self.token2id)