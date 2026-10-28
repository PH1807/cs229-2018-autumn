import torch
x, y = torch.tensor(2.0), torch.tensor(0.0)
w1 = torch.tensor(1.0, requires_grad=True)
w2 = torch.tensor(0.5, requires_grad=True)
y_hat = w2 * torch.relu(w1 * x)          # Forward
J = 0.5 * (y_hat - y) ** 2
J.backward()                              # Backprop = Kettenregel rueckwaerts
print("J =", J.item(), "| dJ/dw1 =", w1.grad.item(), "| dJ/dw2 =", w2.grad.item())
