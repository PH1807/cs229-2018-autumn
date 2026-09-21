import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso, Ridge, LinearRegression
X, y, w = make_regression(100, 20, n_informative=4, noise=10,
                          coef=True, random_state=0)
print("wahre Nicht-Null-Koeff.:", int((w != 0).sum()))
for name, m in [("OLS", LinearRegression()), ("Ridge a=50", Ridge(50)),
                ("Lasso a=2", Lasso(2))]:
    m.fit(X, y)
    print(f"{name:10s} |w|_2 {np.linalg.norm(m.coef_):6.1f}  "
          f"Nullen {int((np.abs(m.coef_) < 1e-8).sum()):2d}")
