import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
rng = np.random.default_rng(0)
X = rng.uniform(0, 10, (200, 1))
y = 3 * X[:, 0] + 2 + rng.normal(0, 1, 200)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
for m in (LinearRegression(), Ridge(alpha=10.0)):
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    print(type(m).__name__, m.coef_.round(2), round(m.intercept_, 2),
          "MSE", round(mean_squared_error(yte, p), 2), "R2", round(r2_score(yte, p), 3))
