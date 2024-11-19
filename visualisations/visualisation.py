# Functions for MDS/t-SNE and plotting

import matplotlib.pyplot as plt
from sklearn.manifold import MDS, TSNE
import seaborn as sns

def plot_rdm(rdm, title, method="MDS"):
    if method == "MDS":
        model = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    elif method == "t-SNE":
        model = TSNE(n_components=2, metric="precomputed", random_state=42, perplexity=3, init="random")

    points_2d = model.fit_transform(rdm)
    plt.figure(figsize=(8, 6))
    plt.scatter(points_2d[:, 0], points_2d[:, 1], s=100, edgecolor="k")
    plt.title(title)
    plt.show()
