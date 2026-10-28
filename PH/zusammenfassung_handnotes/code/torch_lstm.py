import torch, torch.nn as nn
class SeqClassifier(nn.Module):
    def __init__(self, d_in=5, hidden=16, n_cls=3):
        super().__init__()
        self.lstm = nn.LSTM(d_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_cls)
    def forward(self, x):                    # x: (Batch, Zeit, Features)
        out, (h, c) = self.lstm(x)           # h: letzter versteckter Zustand
        return self.head(h[-1])
net = SeqClassifier()
x = torch.randn(8, 20, 5)                    # 8 Sequenzen, 20 Zeitschritte
print("Ausgabe:", tuple(net(x).shape), "| Parameter:", sum(p.numel() for p in net.parameters()))
