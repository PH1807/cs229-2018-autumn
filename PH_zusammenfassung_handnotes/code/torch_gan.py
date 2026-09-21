import torch, torch.nn as nn
torch.manual_seed(0)
G = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))          # Generator
D = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))          # Diskriminator
og, od = torch.optim.Adam(G.parameters(), 1e-3), torch.optim.Adam(D.parameters(), 1e-3)
bce = nn.BCEWithLogitsLoss()
for step in range(3001):
    real = torch.randn(128, 1) * 0.5 + 3.0                                 # Ziel: N(3, 0.5^2)
    fake = G(torch.randn(128, 4))
    ld = bce(D(real), torch.ones(128, 1)) + bce(D(fake.detach()), torch.zeros(128, 1))
    od.zero_grad(); ld.backward(); od.step()                               # D: echt vs. fake
    lg = bce(D(fake), torch.ones(128, 1))                                  # G: D taeuschen
    og.zero_grad(); lg.backward(); og.step()
    if step % 1000 == 0:
        s = G(torch.randn(1000, 4)).detach()
        print(f"step {step:4d}  Mittel {s.mean():.2f}  Std {s.std():.2f}")
