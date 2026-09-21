import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
rng = np.random.default_rng(0); n = 300
df = pd.DataFrame({"alter": rng.integers(18, 70, n).astype(float),
                   "einkommen": rng.normal(50, 15, n),
                   "stadt": rng.choice(["Delhi", "Mumbai", "Pune"], n)})
df.loc[rng.choice(n, 30, replace=False), "einkommen"] = np.nan   # fehlende Werte
y = (df["einkommen"].fillna(50) + rng.normal(0, 10, n) > 50).astype(int)
pre = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                           ("sc", StandardScaler())]), ["alter", "einkommen"]),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), ["stadt"])])
pipe = Pipeline([("pre", pre), ("rf", RandomForestClassifier(random_state=0))])  # kein Leakage
grid = {"rf__n_estimators": [50, 200], "rf__max_depth": [3, None]}
inner = GridSearchCV(pipe, grid, cv=StratifiedKFold(3, shuffle=True, random_state=0))
print("Nested-CV Accuracy:", cross_val_score(inner, df, y, cv=5).mean().round(3))
