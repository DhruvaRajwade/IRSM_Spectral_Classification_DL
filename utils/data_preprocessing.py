from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import torch


def preprocess_data(file_path, scale=True, filter=True, split_ratio=[0.6, 0.2, 0.2], use_torch=False, one_hot=False, num_classes=2):
    """
    Preprocesses data prior to doing ML modelling.

    Parameters:
    file_path (str): Path to the csv file containing the data.
    scale (bool, optional): Whether to apply scaling. Defaults to True.
    filter (bool, optional): Whether to apply Savitzky-Golay filter. Defaults to True.
    split_ratio (list, optional): List of floats representing the train, validation, and test split ratios. Defaults to [0.6, 0.2, 0.2].
    use_torch (bool, optional): Whether to convert the data to tensors. Defaults to False.
    one_hot (bool, optional): Whether to convert the labels to one-hot vectors. Defaults to False.
    num_classes (int, optional): Number of classes. Defaults to 2.

    Returns:
    tuple: A tuple containing the preprocessed training data, validation data, testing data, training labels, validation labels, and testing labels.
    """
    df = pd.read_csv(file_path)
    labels = np.zeros(df.shape[0])  # Labels for control, ,repeat the same for the test CSV
    wavenums = df['Wavenumber']
    data = df.iloc[:, 1:].values  # Update the column indexing

    # Apply scaling if requested
    if scale:
        steps = [('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler(
        )), ('passthrough', FunctionTransformer(func=lambda x: x))]
    else:
        steps = [('imputer', SimpleImputer(strategy='mean')),
                 ('passthrough', FunctionTransformer(func=lambda x: x))]

    # Apply Savitzky-Golay filter if requested
    if filter:
        steps.insert(1, ('filter', FunctionTransformer(
            func=lambda x: savgol_filter(x, window_length=11, polyorder=2, axis=0))))

    # Create the pipeline and apply it to the data
    pipeline = Pipeline(steps)
    X_preprocessed = pipeline.fit_transform(data)

    # Split the data into train, test, and validation sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_preprocessed, labels, test_size=split_ratio[2], random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=split_ratio[1]/(split_ratio[0]+split_ratio[1]), random_state=42)

    if use_torch is True:
        X_train = np.vstack(X_train).astype(float)
        X_test = np.vstack(X_test).astype(float)

        # Convert X_train and X_test to tensors
        X_train = torch.from_numpy(X_train).float()
        X_test = torch.from_numpy(X_test).float()

        # Convert y_train and y_test to tensors
        y_train = torch.from_numpy(y_train).long()
        y_test = torch.from_numpy(y_test).long()
        y_train = torch.nn.functional.one_hot(y_train, num_classes)
        y_test = torch.nn.functional.one_hot(y_test, num_classes)
        y_train = y_train.float()
        y_test = y_test.float()

    print(f'X_train shape: {X_train.shape}', y_train.shape)
    return X_train, X_val, X_test, y_train, y_val, y_test



def pca_transform(X_train, X_test, plot_scree=False):
    """
    Perform PCA on the training data and transform both the training and testing data using the final PCA.

    Parameters:
    X_train (array-like): Training data to fit PCA on.
    X_test (array-like): Testing data to transform using the final PCA.
    plot_scree (bool, optional): Whether to plot the scree plot. Defaults to False.

    Returns:
    tuple: A tuple containing the transformed training and testing data.

    """
    # Fit PCA on training data
    pca = PCA()
    pca.fit(X_train)

    # Plot scree plot if requested
    if plot_scree:
        
        fig, ax = plt.subplots()
        ax.plot(np.arange(1, pca.n_components_+1),
                pca.explained_variance_ratio_, marker='o')
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('Scree Plot')
        plt.show()

    # Determine number of components to use
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1

    # Perform final PCA on training and testing data
    pca_final = PCA(n_components= n_components)
    print(f"N components: {n_components}")
    X_train= pca_final.fit_transform(X_train)
    X_test = pca_final.transform(X_test)

    return X_train, X_test


