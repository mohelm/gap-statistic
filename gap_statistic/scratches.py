import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

make_blob_params = dict(n_samples=100, n_features=2, centers=5, random_state=10)

cm = plt.get_cmap("gist_rainbow")


x, y = make_blobs(**make_blob_params)
x_min = x.min(axis=0)
x_max = x.max(axis=0)

sample = np.random.uniform(x_min, x_max, size=x.shape)


# Plot
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_prop_cycle(
    color=[
        cm(1.0 * i / make_blob_params["centers"])
        for i in range(make_blob_params["centers"])
    ]
)
for i, c in enumerate(range(make_blob_params["centers"])):
    ax1.scatter(x[y == c, 0], x[y == c, 1])
ax1.set_xlim(x_min[0], x_max[0])
ax1.set_ylim(x_min[1], x_max[1])

ax2.scatter(*sample.T)
ax2.set_xlim(x_min[0], x_max[0])
ax2.set_ylim(x_min[1], x_max[1])
plt.show()

from sklearn.metrics.pairwise import euclidean_distances


# Pairwise distances
def sum_of_pairwise_distances(x: np.array) -> int:
    distances = euclidean_distances(x)
    distances[np.triu_indices(len(distances))] = 0
    return distances.sum().sum()


def total_distance(x: np.array, y: np.array) -> int:
    """
    Compute Wk in the paper.

    x: all points
    y: contains the cluster information

    """
    return sum(
        [
            1 / (2 * ((y == c).sum())) * sum_of_pairwise_distances(x[y == c])
            for c in np.unique(y)
        ]
    )


max_clusters = 20
n_samples = 10
random_seed = 55
logWk = []
ELogWk = []
mb = MiniBatchKMeans()
for n_clusters in np.arange(1, max_clusters + 1):
    y = mb.set_params(n_clusters=n_clusters, random_state=random_seed).fit_predict(x)
    logWk.append(np.log(total_distance(x, y)))
    ewk = 0
    for _ in np.arange(n_samples):
        sample = np.random.uniform(x_min, x_max, size=x.shape)
        y = mb.set_params(n_clusters=n_clusters, random_state=random_seed).fit_predict(
            sample
        )
        ewk += (1 / n_samples) * np.log((total_distance(sample, y)))
    ELogWk.append(ewk)

gap = np.array(ELogWk) - np.array(logWk)

f, axs = plt.subplots(2, 2)
axs[0][0].scatter(x[:, 0], x[:, 1])
axs[1][0].plot(np.arange(1, max_clusters + 1), logWk, label="Wk")
axs[1][0].plot(np.arange(1, max_clusters + 1), ELogWk, label="EWk")
axs[1][0].set_xticks(np.arange(1, max_clusters + 1))
axs[1][0].axvline(make_blob_params["centers"], c="r", ls="--")
axs[1][0].legend()
axs[1][1].plot(np.arange(1, max_clusters + 1), gap, label="gap")
axs[1][1].set_xticks(np.arange(1, max_clusters + 1))
axs[1][1].axvline(make_blob_params["centers"], c="r", ls="--")
axs[1][1].legend()
