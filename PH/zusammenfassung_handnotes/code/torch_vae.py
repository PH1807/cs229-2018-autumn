import torch, torch.nn as nn
torch.manual_seed(0)
X = torch.randn(512, 4) @ torch.randn(4, 8) * 0.5           # Daten in 4D-Unterraum
enc = nn.Linear(8, 2 * 2)                                    # -> (mu, log sigma^2), k=2
dec = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 8))
opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=1e-2)
for step in range(301):
    mu, logvar = enc(X).chunk(2, dim=1)
    z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)  # Reparametrisierungs-Trick
    rec = ((dec(z) - X) ** 2).sum(1).mean()                  # -log p(x|z) (Gauss)
    kl = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)).mean()  # KL(Q||N(0,I))
    loss = rec + kl                                          # = -ELBO
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0: print(f"step {step:3d}  rec {rec.item():.3f}  KL {kl.item():.3f}")
