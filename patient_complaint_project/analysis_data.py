import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset import load_json, clean_text, LABEL_KEYS

def analyze_dataset(train_path, test_path, out_dir='./analysis'):
    os.makedirs(out_dir, exist_ok=True)
    train = load_json(train_path)
    test = load_json(test_path)
    def stats(arr):
        ages = [a.get('age') for a in arr if a.get('age') is not None]
        genders = [a.get('gender') for a in arr if a.get('gender') is not None]
        visit_nums = [a.get('visit_num',0) for a in arr]
        texts = [clean_text(a.get('visit_sn','') or '') for a in arr]
        text_lens = [len(t) for t in texts]
        label_counts = {k:0 for k in LABEL_KEYS}
        for a in arr:
            labs = a.get('labels', {})
            for k in LABEL_KEYS:
                if labs.get(k,0)==1:
                    label_counts[k]+=1
        return {'n': len(arr), 'age_mean': float(np.mean(ages)) if ages else None, 'gender_counts': dict(pd.Series(genders).value_counts()), 'visit_num_mean': float(np.mean(visit_nums)) if visit_nums else None, 'text_len_mean': float(np.mean(text_lens)) if text_lens else None, 'text_len_p95': int(np.percentile(text_lens,95)), 'label_counts': label_counts, 'texts': texts}
    tr = stats(train); te = stats(test)
    # save csvs
    pd.DataFrame([tr['label_counts']]).T.rename(columns={0:'train_count'}).to_csv(os.path.join(out_dir,'train_label_counts.csv'))
    pd.DataFrame([te['label_counts']]).T.rename(columns={0:'test_count'}).to_csv(os.path.join(out_dir,'test_label_counts.csv'))
    # text length hist
    plt.figure()
    plt.hist([len(t) for t in tr['texts']], bins=50)
    plt.title('train text length')
    plt.savefig(os.path.join(out_dir,'train_text_len_hist.png')); plt.close()
    plt.figure()
    plt.hist([len(t) for t in te['texts']], bins=50)
    plt.title('test text length')
    plt.savefig(os.path.join(out_dir,'test_text_len_hist.png')); plt.close()
    print('Saved analysis to', out_dir)