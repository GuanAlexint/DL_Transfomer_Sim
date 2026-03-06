import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from model_simcse import SimCSEWrapper
from dataset import ComplaintDataset

def train_simcse(train_arr, tokenizer, out_path, epochs=3, batch_size=32, max_len=256, device='cuda'):
    ds = ComplaintDataset(train_arr, tokenizer, max_len=max_len, stage='unsupervised', augment=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=lambda x: batch_collate(x, tokenizer))
    model = SimCSEWrapper(vocab_size=tokenizer.vocab_size(), d_model=256, num_layers=4)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=2e-4)
    for ep in range(epochs):
        model.train()
        losses=[]
        for batch in dl:
            input_ids = batch['input_ids'].to(device)
            # forward twice (dropout) to get z1 and z2
            z1 = model.encode(input_ids)
            z2 = model.encode(input_ids)
            loss = model.nt_xent_loss(z1, z2)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
        print(f"SimCSE epoch {ep} loss {sum(losses)/len(losses):.4f}")
    torch.save(model.state_dict(), out_path)

# batch collate used to build tensors
def batch_collate(samples, tokenizer):
    import torch
    input_ids = torch.stack([torch.tensor(s['input_ids'], dtype=torch.long) for s in samples], dim=0)
    return {'input_ids': input_ids}