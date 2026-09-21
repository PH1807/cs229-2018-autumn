from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
X, _ = make_blobs(300, centers=3, cluster_std=0.8, random_state=0)
for k in (2, 3, 4, 5):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    print(f"k={k} Inertia={km.inertia_:7.1f} Silhouette={silhouette_score(X, km.labels_):.3f}")
agg = AgglomerativeClustering(n_clusters=3, linkage="ward").fit(X)
gmm = GaussianMixture(n_components=3, random_state=0).fit(X)
print("Hierarchisch Clustergroessen:", sorted(map(int, __import__("numpy").bincount(agg.labels_))))
print("GMM Gewichte phi:", gmm.weights_.round(2))
