import torch
import torch.nn as nn
import torch.nn.functional as F

from model_transformer import TransformerEncoderModel

class SimCSEWrapper(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=4, dim_feedforward=512):
        super().__init__()
        # reuse encoder but without classifier head usage
        # we instantiate same architecture but will only use pooled output
        self.encoder = TransformerEncoderModel(vocab_size, d_model=d_model, num_heads=num_heads, num_layers=num_layers, dim_feedforward=dim_feedforward, struct_dim=10)
        self.temperature = 0.05

    def encode(self, input_ids):
        # return pooled vector before classifier
        device = input_ids.device
        batch_size = input_ids.size(0)
        # create dummy struct zeros
        struct = torch.zeros((batch_size,10), device=device)
        # forward through encoder to get pooled vector by leveraging classifier input
        with torch.no_grad():
            out = self.encoder(input_ids, (input_ids!=0).long(), struct)
        # We don't have direct access to pooled inside encoder here; so ideally split encoder.
        # For simplicity in this wrapper, assume encoder.embed+transformer accessible:
        x = self.encoder.embed(input_ids) * (self.encoder.embed.embedding_dim ** 0.5)
        x = self.encoder.pos(x)
        key_padding_mask = (input_ids == 0)
        out_hidden = self.encoder.transformer(x, src_key_padding_mask=key_padding_mask)
        mask = (input_ids != 0).unsqueeze(-1).float()
        summed = (out_hidden * mask).sum(1)
        denom = mask.sum(1).clamp(min=1e-9)
        pooled = summed / denom
        return pooled

    def forward(self, input_ids):
        return self.encode(input_ids)

    def nt_xent_loss(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.matmul(z, z.T) / self.temperature
        # mask out self
        N = z1.size(0)
        labels = torch.arange(N, device=z.device)
        # compute logits for i against j
        logits = torch.cat([sim[:N, N:], sim[N:, :N]], dim=1)
        # targets are diagonal
        targets = torch.arange(N, device=z.device)
        loss = F.cross_entropy(logits, targets)
        return loss