import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from model_transformer import TransformerEncoderModel
from evaluate import compute_metrics

def train_supervised(train_arr, val_arr, tokenizer, out_dir, simcse_ckpt=None, epochs=6, batch_size=16, lr=2e-4, max_len=256, device='cuda'):
    train_ds = create_dataset(train_arr, tokenizer, max_len=max_len, stage='train', augment=True)
    val_ds = create_dataset(val_arr, tokenizer, max_len=max_len, stage='val', augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = TransformerEncoderModel(vocab_size=tokenizer.vocab_size(), d_model=256, num_layers=4)
    if simcse_ckpt:
        try:
            state = torch.load(simcse_ckpt, map_location='cpu')
            model.load_state_dict(state, strict=False)
            print('Loaded simcse checkpoint where possible.')
        except Exception as e:
            print('Failed to load simcse ckpt:', e)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_micro = 0.0
    for ep in range(epochs):
        model.train()
        losses=[]
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            struct = batch['struct'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids, attn, struct)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {ep} train_loss {sum(losses)/len(losses):.4f}")
        metrics = compute_metrics(model, val_loader, device)
        print(f"Val microF1 {metrics['micro_f1']:.4f} macroF1 {metrics['macro_f1']:.4f}")
        if metrics['micro_f1'] > best_micro:
            best_micro = metrics['micro_f1']
            torch.save(model.state_dict(), f"{out_dir}/best_supervised.pth")
    print('Best micro-F1', best_micro)

# helper to create dataset (avoid circular import)
from dataset import ComplaintDataset
def create_dataset(arr, tokenizer, max_len=256, stage='train', augment=False):
    ds = ComplaintDataset(arr, tokenizer, max_len=max_len, stage=stage, augment=augment)
    # collate to dicts
    class Wrapper(torch.utils.data.Dataset):
        def __len__(self): return len(ds)
        def __getitem__(self, idx):
            it = ds[idx]
            return {'input_ids': it['input_ids'], 'attention_mask': it['attention_mask'], 'struct': it['struct'], 'labels': it['labels']}
    return Wrapper()