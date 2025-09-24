import torch, torch.nn as nn

class TransformerClassifier(nn.Module):
    def __init__(self, D, hidden=64, n_classes=2, n_heads=4, n_layers=2):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=D, nhead=n_heads, dim_feedforward=hidden)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.fc = nn.Linear(D, n_classes)

    def forward(self, x):
        # x: (B,T,D) -> (T,B,D) for transformer
        h = self.transformer(x.transpose(0,1))
        h = h.mean(0)
        return self.fc(h)
