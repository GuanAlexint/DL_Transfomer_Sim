import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerEncoderModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=4, dim_feedforward=512, struct_dim=10, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.struct_mlp = nn.Sequential(nn.Linear(struct_dim, 64), nn.ReLU(), nn.Linear(64, 32))
        self.classifier = nn.Sequential(nn.Linear(d_model+32, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 9))

    def forward(self, input_ids, attention_mask, struct_feats):
        # input_ids: (B, L)
        x = self.embed(input_ids) * math.sqrt(self.embed.embedding_dim)
        x = self.pos(x)
        # transformer (batch_first=True)
        # create key_padding_mask: True where padding (to be masked)
        key_padding_mask = (input_ids == 0)
        out = self.transformer(x, src_key_padding_mask=key_padding_mask)
        # mean pool over non-pad tokens
        mask = (input_ids != 0).unsqueeze(-1).float()
        summed = (out * mask).sum(1)
        denom = mask.sum(1).clamp(min=1e-9)
        pooled = summed / denom
        struct_emb = self.struct_mlp(struct_feats)
        cat = torch.cat([pooled, struct_emb], dim=1)
        logits = self.classifier(cat)
        return logits