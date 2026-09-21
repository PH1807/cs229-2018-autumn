import numpy as np
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, mean_absolute_error,
                             mean_squared_error, r2_score)
y = np.array([1]*10 + [0]*10)
s = np.array([.9,.8,.85,.7,.95,.6,.75,.9,.4,.55, .3,.2,.6,.1,.45,.35,.65,.25,.15,.05])
p = (s >= 0.5).astype(int)
print(confusion_matrix(y, p).tolist(), "[[TN,FP],[FN,TP]]")
print("P", precision_score(y, p), "R", recall_score(y, p), "F1", round(f1_score(y, p), 3))
print("AUROC", round(roc_auc_score(y, s), 3), "AUPRC", round(average_precision_score(y, s), 3))
yt = np.array([5.0, 6.0, 7.5, 8.0, 9.5]); yp = np.array([4.6, 5.5, 8.0, 7.2, 10.0])
print("MAE", mean_absolute_error(yt, yp), "MSE", round(mean_squared_error(yt, yp), 3),
      "RMSE", round(mean_squared_error(yt, yp) ** .5, 3), "R2", round(r2_score(yt, yp), 3))
