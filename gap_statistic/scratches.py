import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

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
    ax1.scatter(x[y == c, 0], x[y==c, 1])
ax1.set_xlim(x_min[0], x_max[0])
ax1.set_ylim(x_min[1], x_max[1])

ax2.scatter(*sample.T)
ax2.set_xlim(x_min[0], x_max[0])
ax2.set_ylim(x_min[1], x_max[1])
plt.show()
