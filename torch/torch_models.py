import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


# Define the VAE architecture


class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, latent_size)
        self.fc3 = nn.Linear(hidden_size, latent_size)
        self.fc4 = nn.Linear(latent_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, input_size)

    def encoder(self, x):
        out = torch.relu(self.fc1(x))
        return self.fc2(out), self.fc3(out)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decoder(self, z):
        out = torch.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(out))

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


class ARN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ARN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.attention = nn.Linear(input_size, hidden_size)
        self.residual_1 = nn.Linear(input_size, hidden_size)
        self.residual_2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        attention_out = self.attention(x)
        residual_out = self.residual_1(x)
        x = attention_out + residual_out
        x = F.relu(x)
        residual_out = self.residual_2(x)
        x = x + residual_out
        x = F.relu(x)
        x = self.output(x)
        return x
