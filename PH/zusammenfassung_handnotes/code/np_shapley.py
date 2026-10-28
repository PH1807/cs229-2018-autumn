from itertools import permutations
import numpy as np
# Modell mit Interaktion: f(x) = 3*x1 + 2*x2 + 4*x1*x2
f = lambda x: 3*x[0] + 2*x[1] + 4*x[0]*x[1]
x, base = np.array([1., 1.]), np.array([0., 0.])   # Instanz, Baseline
def v(S):                       # Koalition: nur Features in S bekannt
    z = base.copy()
    for j in S: z[j] = x[j]
    return f(z)
phi, perms = np.zeros(2), list(permutations(range(2)))
for p in perms:                 # Mittel ueber alle Reihenfolgen
    S = []
    for j in p:
        phi[j] += (v(S + [j]) - v(S)) / len(perms)
        S.append(j)
print("phi   =", phi)
print("Summe =", phi.sum(), "; f(x)-f(0) =", f(x) - f(base))
