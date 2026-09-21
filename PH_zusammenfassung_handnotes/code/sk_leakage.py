import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
rng = np.random.default_rng(0)
X = rng.normal(size=(100, 1000))            # reines Rauschen
y = rng.integers(0, 2, 100)                 # zufaellige Labels
clf = LogisticRegression(max_iter=1000)
Xsel = SelectKBest(f_classif, k=20).fit_transform(X, y)   # LEAK: alle Daten
print("Leakage  :", cross_val_score(clf, Xsel, y, cv=5).mean().round(2))
pipe = make_pipeline(SelectKBest(f_classif, k=20), clf)   # Auswahl im Fold
print("Pipeline :", cross_val_score(pipe, X, y, cv=5).mean().round(2))
