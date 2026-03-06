import json
import random
import torch
from torch.utils.data import Dataset
from tokenizer import Tokenizer

LABEL_KEYS = ['tired','sleep difficulty','appetite decreased','move slowly',
              'irritable','cognitive','weight decreased','weight increased','dispirited']

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        txt = f.read().strip()
    try:
        data = json.loads(txt)
        if isinstance(data, list):
            return data
    except:
        pass
    arr = []
    for line in txt.splitlines():
        line = line.strip()
        if not line: continue
        try:
            arr.append(json.loads(line))
        except:
            continue
    return arr

def clean_text(s):
    if not s: return ''
    s = s.replace('\r',' ').replace('\n',' ').strip()
    s = ' '.join(s.split())
    s = s.replace('否认14天内','')
    return s

def build_struct(item):
    age = float(item.get('age',0) or 0)/100.0
    visit_num = float(item.get('visit_num',0))/50.0
    gender = float(item.get('gender',0) or 0)
    flags = ['is_hypertension','is_ischaemic_heart','is_heart_failure','is_renal','is_pad','is_dementia','is_cvd']
    vals = [float(item.get(k,0) or 0) for k in flags]
    arr = [age, visit_num, gender] + vals
    return torch.tensor(arr, dtype=torch.float32)

class ComplaintDataset(Dataset):
    def __init__(self, arr, tokenizer: Tokenizer, max_len=256, stage='train', augment=False):
        self.arr = arr
        self.tok = tokenizer
        self.max_len = max_len
        self.stage = stage
        self.augment = augment

    def augment_text(self, text):
        # simple augmentations: synonym replacement & random deletion
        # user can extend SYN dict
        SYN = {'脑梗死':['脑梗塞','中风'],'乏力':['无力','疲倦'],'言语不清':['言语含糊','说话不利索']}
        if random.random() < 0.2:
            for k,vs in SYN.items():
                if k in text and random.random() < 0.5:
                    text = text.replace(k, random.choice(vs))
        if random.random() < 0.1:
            toks = text.split()
            if len(toks) > 3:
                text = ' '.join([w for w in toks if random.random() > 0.1])
        return text

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, idx):
        item = self.arr[idx]
        raw = item.get('visit_sn','') or ''
        text = clean_text(raw)
        if self.stage == 'train' and self.augment:
            text = self.augment_text(text)
        ids = self.tok.encode(text, max_len=self.max_len)
        input_ids = torch.tensor(ids, dtype=torch.long)
        attn_mask = (input_ids != self.tok.token2id['[PAD]']).long()
        struct = build_struct(item)
        labels = torch.tensor([item.get('labels',{}).get(k,0) for k in LABEL_KEYS], dtype=torch.float32)
        return {'input_ids': input_ids, 'attention_mask': attn_mask, 'struct': struct, 'labels': labels}