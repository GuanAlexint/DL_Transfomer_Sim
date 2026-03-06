import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_recall_fscore_support

def compute_metrics(model, dataloader, device='cuda'):
    model.eval()
    preds=[]; probs=[]; trues=[]
    with __import__('torch').no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            struct = batch['struct'].to(device)
            labels = batch['labels'].cpu().numpy()
            logits = model(input_ids, attn, struct)
            prob = __import__('torch').sigmoid(logits).cpu().numpy()
            preds.append((prob>=0.5).astype(int))
            probs.append(prob)
            trues.append(labels)
    y_true = np.vstack(trues)
    y_prob = np.vstack(probs)
    y_pred = np.vstack(preds)
    micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    aucs=[]
    aps=[]
    for i in range(y_true.shape[1]):
        try:
            aucs.append(float(roc_auc_score(y_true[:,i], y_prob[:,i])))
        except:
            aucs.append(float('nan'))
        try:
            aps.append(float(average_precision_score(y_true[:,i], y_prob[:,i])))
        except:
            aps.append(float('nan'))
    return {'micro_f1': micro, 'macro_f1': macro, 'aucs': aucs, 'aps': aps}