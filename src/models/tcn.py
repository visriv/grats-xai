import torch, torch.nn as nn

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k,
                              dilation=dilation, padding=(k-1)*dilation)
        self.relu = nn.ReLU()
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.relu(self.conv(x))
        if self.down: x = self.down(x)
        return out[:, :, :x.size(2)] + x

class TCNClassifier(nn.Module):
    def __init__(self, D, hidden=64, n_classes=2):
        super().__init__()
        self.tcn = nn.Sequential(
            TCNBlock(D, hidden, k=3, dilation=1),
            TCNBlock(hidden, hidden, k=3, dilation=2)
        )
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):
        # x = x.transpose(1,2)  # (B,T,D) -> (B,D,T)
        h = self.tcn(x).mean(-1)
        return self.fc(h)
