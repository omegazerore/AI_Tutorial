import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    fowlkes_mallows_score, homogeneity_score, completeness_score,
    v_measure_score, confusion_matrix, accuracy_score
)
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA

# --------------------------
# Step 1: Load Data
# --------------------------
iris = load_iris()
X = iris.data
y_true = iris.target

# --------------------------
# Step 2: Fit Clustering Model
# --------------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)


def map_clusters(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    cost_matrix = np.zeros((D, D), dtype=np.int64)
    for i in range(len(y_true)):
        cost_matrix[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    mapping = {r: c for r, c in zip(row_ind, col_ind)}
    return np.array([mapping[cluster] for cluster in y_pred])

y_pred_mapped = map_clusters(y_true, y_pred)