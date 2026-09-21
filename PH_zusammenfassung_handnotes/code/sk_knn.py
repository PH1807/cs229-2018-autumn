from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
X, y = load_iris(return_X_y=True)
for k in (1, 5, 15, 45):
    m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    print("k =", k, " CV-Acc:", cross_val_score(m, X, y, cv=5).mean().round(3))
