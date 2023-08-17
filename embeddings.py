from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


def center_and_scale(data):
    scaler= StandardScaler(with_mean=True, with_std=True)
    return scaler.fit_transform(data)

def PCA_images(data, cumulative_percentage=0.8):
    pca = PCA()

    pca.fit(data)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explainedª Variance Ratio')
    plt.title('Cumulative Explained Variance Ratio vs. Number of PCs')
    plt.grid(True)
    plt.show()

    components = np.argmax(cumulative_variance >= cumulative_percentage) + 1
    print("Number of components necessary for ", cumulative_percentage*100, "%% cumulative explained percentage is ", components)

    pca = PCA(n_components=components)
    reduced_data = pca.fit_transform(data)
    # Verify the shape of the reduced_data
    print("Shape of reduced_data:", reduced_data.shape)

    return reduced_data
