from sklearn.model_selection import KFold
import numpy as np
import torch

def cross_validate(model, X, y, num_folds, num_epochs, optimizer, loss_fn):
    """
    Perform cross-validation.

    This function performs k-fold cross-validation using the specified
    data (X and y) and hyperparameters (num_folds, num_epochs, optimizer, loss_fn). It trains
    and evaluates the model on each fold, returning the mean accuracy and standard deviation
    of the accuracies across folds.

    Args:
        model (torch.nn.Module): A PyTorch model.
        X (numpy.ndarray): Input data.
        y (numpy.ndarray): Target labels.
        num_folds (int): Number of folds for cross-validation.
        num_epochs (int): Number of epochs for training.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        loss_fn (torch.nn.Module): Loss function for model training.

    Returns:
        float: Mean accuracy across folds.
        float: Standard deviation of accuracies across folds.
    """
    # Define the KFold object
    kf = KFold(n_splits=num_folds, shuffle=True)

    # Initialize a list to store the results
    results = []

    # Loop over the folds
    for train_index, test_index in kf.split(X):
        # Get the training and test data for this fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train = np.vstack(X_train).astype(float)
        X_test = np.vstack(X_test).astype(float)

        # Convert the data to tensors
        X_train = torch.Tensor(X_train)
        y_train = torch.Tensor(y_train).long()
        X_test = torch.Tensor(X_test)
        y_test = torch.Tensor(y_test).long()
        y_train = torch.nn.functional.one_hot(y_train, num_classes=2)
        y_test = torch.nn.functional.one_hot(y_test, num_classes=2)
        y_train = y_train.float()
        y_test = y_test.float()

        # Loop over the epochs for training
        for epoch in range(num_epochs):
            # Zero out the gradients of the model's parameters
            optimizer.zero_grad()

            # Pass the training data through the model to get the predictions
            y_pred = model(X_train)

            # Calculate the loss using the loss function
            loss = loss_fn(y_pred, y_train)

            # Backpropagate the error to compute the gradients of the model's parameters
            loss.backward()

            # Update the model's parameters using the optimizer
            optimizer.step()

        # Convert the predicted probabilities to class labels
        y_pred = (y_pred > 0.5).long()

        # Calculate the accuracy
        accuracy = (y_pred == y_train).float().mean()

        # Append the accuracy to the results list
        results.append(accuracy.item())

    # Calculate mean and standard deviation of accuracies across folds
    mean = np.mean(results)
    std = np.std(results)

    return mean, std


# Example usage
"""
X = X_train_latent
y = y_train

# Assuming you have defined the model, optimizer, and loss_fn variables
num_folds = 10
num_epochs = 100

mean_accuracy, std_accuracy = cross_validate(model, X, y, num_folds, num_epochs, optimizer, loss_fn)

print(f"Mean accuracy: {mean_accuracy}")
print(f"Standard deviation: {std_accuracy}")
"""