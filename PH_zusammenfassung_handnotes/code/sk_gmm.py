import numpy as np
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
X, _ = make_blobs(300, centers=3, cluster_std=[1.0, 1.5, 0.7],
                  random_state=0)
prev = -np.inf
for it in (1, 2, 3, 5, 10, 20):             # EM-Iterationen fortsetzen
    gmm = GaussianMixture(3, max_iter=it, tol=0, init_params="random",
                          random_state=1).fit(X)
    ll = gmm.score(X) * len(X)              # Log-Likelihood
    print(f"iter {it:2d}  logL {ll:8.1f}  steigt: {ll >= prev - 1e-6}")
    prev = ll
