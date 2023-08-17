from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import cv2
import numpy as np


def tsne(data, labels):
    tsne = TSNE(n_components=2)
    data_tsne = tsne.fit_transform(data)

    unique_labels = np.unique(labels) 

    plt.figure(figsize=(8, 6))
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        plt.scatter(data_tsne[indices, 0], data_tsne[indices, 1], label=f'Cluster {label}', alpha=0.6)

    plt.title('Agglomerative Clustering with t-SNE Visualization')
    #plt.legend()
    plt.show()


def show_cluster_examples(video_path, cluster_labels, cluster, num_examples):
    cap = cv2.VideoCapture(video_path)
    frames_to_show = np.random.choice(cluster_labels[cluster][:100], num_examples, replace=False)
    print(frames_to_show)
    fig, axs = plt.subplots(1, num_examples, figsize=(20, 15))
    #fig.suptitle(f"Cluster {cluster}")
    
    for i in range(1, num_examples+1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames_to_show[i-1])
        res, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR to RGB
        axs[i-1].imshow(frame)
        axs[i-1].axis('off')
        #plt.show()

