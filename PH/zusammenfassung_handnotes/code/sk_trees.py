from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
X, y = load_wine(return_X_y=True)
dt = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0)
rf = RandomForestClassifier(n_estimators=200, max_features="sqrt", random_state=0)
print("Baum  CV-Acc:", cross_val_score(dt, X, y, cv=5).mean().round(3))
print("Wald  CV-Acc:", cross_val_score(rf, X, y, cv=5).mean().round(3))
rf.fit(X, y)
print("Top-3 Feature-Importance:", rf.feature_importances_.argsort()[::-1][:3])
