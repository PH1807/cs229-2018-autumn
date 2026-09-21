from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
clf.fit(Xtr, ytr)                              # C = 1/lambda (L2)
print("Accuracy:", round(clf.score(Xte, yte), 3))
print("P(y=1|x) der ersten 3:", clf.predict_proba(Xte[:3])[:, 1].round(3))
