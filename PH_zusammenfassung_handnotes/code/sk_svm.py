from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
X, y = make_moons(300, noise=0.25, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
grid = {"C": [0.1, 1, 10], "gamma": [0.1, 1, 10]}
gs = GridSearchCV(SVC(kernel="rbf"), grid, cv=5).fit(Xtr, ytr)
print("beste Parameter:", gs.best_params_)
print("Test-Accuracy  :", round(gs.score(Xte, yte), 3))
print("Support Vectors:", gs.best_estimator_.n_support_)
