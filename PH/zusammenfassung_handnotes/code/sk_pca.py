from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
X, _ = load_digits(return_X_y=True)
Z = StandardScaler().fit_transform(X)
pca = PCA().fit(Z)
cev = pca.explained_variance_ratio_.cumsum()
print("Komponenten fuer 90% Varianz:", int((cev < 0.90).sum() + 1), "von", X.shape[1])
print("erste 3 EVR:", pca.explained_variance_ratio_[:3].round(3))
