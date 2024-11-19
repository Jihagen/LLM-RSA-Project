
from scipy.spatial.distance import pdist, squareform

def compute_rdm(activation_tensor, metric="cosine"):
    act_flat = activation_tensor.view(activation_tensor.size(0), -1).cpu().numpy()
    distances = pdist(act_flat, metric=metric)
    return squareform(distances)
