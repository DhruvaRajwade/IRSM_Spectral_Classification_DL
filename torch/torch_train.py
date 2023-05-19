from torch_models import *
from vae_loss import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def train_vae(X_train, X_test, loss_fn, hidden_size, latent_size, num_epochs):
    """
    Trains a Variational Autoencoder (VAE) model using the given data and hyperparameters.

    Args:
    - X_train (torch.Tensor): The training data tensor.
    - X_test (torch.Tensor): The test data tensor.
    - loss_fn (torch.nn.Module): The loss function to use for training.
    - hidden_size (int): The number of hidden units.
    - latent_size (int): The number of latent variables.
    - num_epochs (int): The number of epochs to train the model.

    Returns:
    - X_train_latent (torch.Tensor): The final latent representation of the training data.
    - X_test_latent (torch.Tensor): The final latent representation of the test data.
    """
 # Initialize the model and optimizer
    input_size = X_train.shape[1]
    model = VAE(input_size, hidden_size, latent_size)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train the model for the specified number of epochs
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        recon_x, mu, logvar = model(X_train)
        loss = loss_fn(recon_x, X_train, mu, logvar)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print("Epoch: {}, Loss: {:.4f}".format(epoch, loss.item()))

    # Extract features from the trained VAE
    model.eval()
    with torch.no_grad():
        X_train_latent = model.encoder(X_train)[0]
        X_test_latent = model.encoder(X_test)[0]

    return X_train_latent, X_test_latent

def train_classifier(X_train_latent, y_train, X_test_latent, y_test, input_dim, hidden_dim, num_classes, num_epochs):
    """
    Trains a neural network model using the given data and hyperparameters.

    Args:
    - X_train_latent (torch.Tensor): The training data tensor.
    - y_train (torch.Tensor): The training labels tensor.
    - X_test_latent (torch.Tensor): The test data tensor.
    - y_test (torch.Tensor): The test labels tensor.
    - input_dim (int): The number of features.
    - hidden_dim (int): The number of hidden units.
    - num_classes (int): The number of classes.
    - num_epochs (int): The number of epochs to train the model.

    Returns:
    - model (torch.nn.Module): The trained neural network model.
    """


    # Initialize the model and loss function
    model = ARN(input_dim, hidden_dim, num_classes)
    loss_fn = nn.BCEWithLogitsLoss()

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # Train the model for the specified number of epochs
    for epoch in range(num_epochs):
        # Zero out the gradients of the model's parameters
        optimizer.zero_grad()

        # Pass the input data through the model to get the predictions
        y_pred = model(X_train_latent)

        # Calculate the loss using the loss function
        loss = loss_fn(y_pred, y_train)

        # Backpropagate the error to compute the gradients of the model's parameters
        loss.backward()

        # Update the model's parameters using the optimizer
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}: loss = {loss.item()}')

    # Test the model on the test data and evaluate its performance
    y_pred = model(X_test_latent)
    loss = loss_fn(y_pred, y_test)
    print(f'Test loss: {loss.item()}')

    # Convert the predicted probabilities to class labels
    y_pred = (y_pred > 0.5).long()

    # Calculate the accuracy
    accuracy = (y_pred == y_test).float().mean()
    print(f'Accuracy: {accuracy.item()}')

    return model
