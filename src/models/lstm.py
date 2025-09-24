
import torch, torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, D, hidden=64, num_layers=1, n_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=D, hidden_size=hidden, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.LayerNorm(2*hidden), nn.ReLU(), nn.Linear(2*hidden, n_classes))
    def forward(self, x): # x: (B, D, T)
        x = x.permute(0,2,1)  # (B,T,D)
        h,_ = self.lstm(x)    # (B,T,2H)
        h = h.mean(dim=1)     # temporal pooling
        return self.head(h)
