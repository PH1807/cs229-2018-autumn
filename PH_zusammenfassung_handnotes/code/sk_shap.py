import numpy as np, shap
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
X, y = load_diabetes(return_X_y=True, as_frame=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
rf = RandomForestRegressor(200, min_samples_leaf=5, random_state=0)
rf.fit(Xtr, ytr)
imp = rf.feature_importances_                    # Impurity (Train)
perm = permutation_importance(rf, Xte, yte, n_repeats=20,
                              random_state=0).importances_mean   # Test
expl = shap.TreeExplainer(rf)
sv = expl.shap_values(Xte)                       # (n, d) Shapley-Werte
glob = np.abs(sv).mean(0)                        # globale Wichtigkeit
rows = sorted(zip(X.columns, imp, perm, glob), key=lambda t: -t[3])
for n, a, b, c in rows[:4]:
    print(f"{n:4s} imp {a:.2f} perm {b:.2f} SHAP {c:.1f}")
pred = rf.predict(Xte)[0]                        # lokal: 1 Patient
print("Additiv.:", round(pred, 1), "=",
      round(expl.expected_value[0] + sv[0].sum(), 1))
