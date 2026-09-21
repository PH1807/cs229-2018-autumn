import torch, torch.nn as nn
from sklearn.datasets import make_moons
torch.manual_seed(0)
X, y = make_moons(400, noise=0.2, random_state=0)
X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y)
model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Dropout(0.2),
                      nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))
opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)  # L2
loss_fn = nn.CrossEntropyLoss()                # Softmax + Cross-Entropy
for epoch in range(200):
    model.train()
    for i in torch.randperm(len(X)).split(64):  # Mini-Batches
        opt.zero_grad()
        loss = loss_fn(model(X[i]), y[i])
        loss.backward()                         # Backprop
        opt.step()                              # Update
    if epoch % 50 == 0:
        model.eval()
        acc = (model(X).argmax(1) == y).float().mean().item()
        print(f"Epoche {epoch:3d}  loss {loss.item():.3f}  acc {acc:.3f}")
