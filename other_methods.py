from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.cluster import SpectralClustering


# VALUE IMPUTATION METHODS

def knn_imputation(data, num_neighbours=5):
    imputer = KNNImputer(n_neighbors=num_neighbours)
    data_knn_imp = imputer.fit_transform(data)
    return data_knn_imp

def mice_imputer(data, max=10):
    imputer = IterativeImputer(random_state=100, max_iter=max)
    data_mice_imp = imputer.fit_transform(data)
    # https://www.machinelearningplus.com/machine-learning/mice-imputation/
    # https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer
    return data_mice_imp

# Clustering

def spectral_clustering(data, number_clusters):
    clustering = SpectralClustering(n_clusters=number_clusters, assign_labels='cluster_qr')
    data_clustered = clustering.fit_predict(data)
    return data