import numpy as np
from fairlearn.metrics import (MetricFrame, selection_rate, true_positive_rate,
    demographic_parity_difference, equalized_odds_difference)
from sklearn.metrics import accuracy_score, precision_score
# Gruppe A (20 Antraege) und Gruppe B (20 Antraege)
y_true = np.array([1]*8 + [0]*12 + [1]*4 + [0]*16)
y_pred = np.array([1]*6 + [0]*2 + [1]*2 + [0]*10
                  + [1]*1 + [0]*3 + [1]*1 + [0]*15)
g = np.array(["A"]*20 + ["B"]*20)
mf = MetricFrame(metrics={"select": selection_rate, "TPR": true_positive_rate,
                          "PPV": precision_score, "acc": accuracy_score},
                 y_true=y_true, y_pred=y_pred, sensitive_features=g)
mf.by_group.index.name = "Gruppe"
print(mf.by_group.round(2))
dp = demographic_parity_difference(y_true, y_pred, sensitive_features=g)
eo = equalized_odds_difference(y_true, y_pred, sensitive_features=g)
print("DP-Diff  :", round(dp, 2))
print("EO-Diff  :", round(eo, 2))
print("Disp.Imp.:", mf.by_group.select["B"] / mf.by_group.select["A"])
