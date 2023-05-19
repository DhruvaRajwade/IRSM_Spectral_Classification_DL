import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA

def visualize_pca(X_train, encodeClassdata):
    """
    Perform PCA on the input data and visualize the results.

    Args:
        X_train (array-like): Training data matrix of shape (n_samples, n_features).
        encodeClassdata (array-like): Class labels or colors corresponding to each sample in X_train.

    Returns:
        None
    """

    # Perform PCA
    pca = PCA()
    pca.fit(X_train)
    PCAmat = pca.transform(X_train)

    # Plot the scree plot
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, pca.n_components_+1), pca.explained_variance_ratio_, marker='o')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('Scree Plot')
    plt.show()

    # Determine the number of components to use
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    print('Number of components to use:', n_components)

    # Create labels for scatter matrix plot
    labels = {
        str(i): f" {i+1} ({var:.1f}%)" for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }

    # Scatter matrix plot
    fig = px.scatter_matrix(
        X_train,
        labels=labels,
        dimensions=range(4),
        color=encodeClassdata[:X_train.shape[0]]
    )
    fig.update_traces(diagonal_visible=False)
    fig.show()

    # Visualize PC1 vs PC2
    plot = plt.scatter(PCAmat[:, 0], PCAmat[:, 1], c=encodeClassdata[:X_train.shape[0]])
    plt.show()

    # Loadings (All)
    loadings = plt.figure()
    for count, row in enumerate(pca.components_):
        plt.plot(labels, np.transpose(pca.components_[count]), label="PC " + str(count+1))
    plt.title("Loadings")
    plt.ylabel("Counts")
    plt.xlabel("Raman Shifts ($cm^{-1}$)")
    plt.legend()
    plt.show()

    # Loadings (Selected)
    for count, row in enumerate(pca.components_[:2]):
        plt.plot(labels, np.transpose(pca.components_[count]), label="PC " + str(count+1))
    plt.title("Loadings")
    plt.ylabel("Loadings")
    plt.xlabel("Wavenumber ($cm^{-1}$)")
    plt.legend()
    plt.show()
