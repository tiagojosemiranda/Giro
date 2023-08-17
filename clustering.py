import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import AgglomerativeClustering


def agglomerative_clustering(data, n, affinity='euclidean', linkage='ward'):
    cluster_object = AgglomerativeClustering(n_clusters=n, affinity=affinity, linkage=linkage, compute_full_tree=True)
    cluster_labels = cluster_object.fit_predict(data)
    return cluster_labels

def dendograma(matrix, met = 'ward'):
    Z = sch.linkage(matrix, method = met)
    plt.figure(figsize=(10, 6))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Indexes')
    plt.ylabel('Distance')
    dendograma = sch.dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
    plt.show()
    return dendograma


def k_means(data, n_max):
    wcss = {} 
    num_cluster = []
    sill = []
    for no_clust in range(2, n_max):
        kmeans = KMeans(n_clusters=no_clust)
        cluster_labels = kmeans.fit_predict(data)
        wcss[no_clust] = (kmeans.inertia_)
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(data, cluster_labels)
        print("For n_clusters =", no_clust, "The average silhouette_score is :", silhouette_avg)
        num_cluster.append(no_clust)
        sill.append(silhouette_avg)
        #print(f'The within cluster sum of squares for {no_clust} clusters is {wcss[no_clust]:.2f}')
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(data, cluster_labels)

    plt.figure(figsize=(8, 6))
    plt.scatter(num_cluster, sill)
    plt.title('Average silhouette score per number of clusters')
    plt.show()
    return wcss
        

def get_clusters_by_frame(cluster_labels, n_clusters): #each cluster might have different number of frames so we don't want to use numpy arrays
    labels = []
    for i in range(n_clusters):
        # In each iteration, add an empty list to the main list
        labels.append([])

    for frame, cluster_assignment in enumerate(cluster_labels):
        labels[cluster_assignment].append(frame)

    return labels