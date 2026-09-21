from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              StackingClassifier, VotingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
X, y = load_breast_cancer(return_X_y=True)
base = [("rf", RandomForestClassifier(100, random_state=0)),
        ("gb", GradientBoostingClassifier(random_state=0))]
models = {"AdaBoost": AdaBoostClassifier(random_state=0),
          "Voting": VotingClassifier(base, voting="soft"),
          "Stacking": StackingClassifier(base, final_estimator=LogisticRegression(max_iter=2000))}
for name, m in models.items():
    print(f"{name:9s} CV-Acc:", cross_val_score(m, X, y, cv=5).mean().round(3))
