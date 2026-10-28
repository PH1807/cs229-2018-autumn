import torch, torch.nn as nn
net = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28 -> 14
    nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14 -> 7
    nn.Flatten(), nn.Linear(16 * 7 * 7, 10))
x = torch.randn(4, 1, 28, 28)                        # Batch von 4 Bildern
print("Ausgabe:", tuple(net(x).shape))
print("Parameter Conv1:", sum(p.numel() for p in net[0].parameters()), "(= 3*3*1*8 + 8)")
print("Parameter gesamt:", sum(p.numel() for p in net.parameters()))
