import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils.data_preprocessing import *


def train_and_test(X_train, y_train, X_test, y_test):
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=True)

    lost = []

    def overlay_y_on_x(x, y):
        x_ = x.clone()
        x_[:, :10] *= 0.0
        x_[range(x.shape[0]), y] = x.max()
        return x_

    class Net(torch.nn.Module):
        def __init__(self, dims):
            super().__init__()
            self.layers = []
            for d in range(len(dims) - 1):
                self.layers += [Layer(dims[d], dims[d + 1]).cuda()]

        def predict(self, x):
            goodness_per_label = []
            for label in range(10):
                h = overlay_y_on_x(x, label)
                goodness = []
                for layer in self.layers:
                    h = layer(h)
                    goodness += [h.pow(2).mean(1)]
                goodness_per_label += [sum(goodness).unsqueeze(1)]
            goodness_per_label = torch.cat(goodness_per_label, 1)
            return goodness_per_label.argmax(1)

        def train(self, x_pos, x_neg):
            h_pos, h_neg = x_pos, x_neg
            for i, layer in enumerate(self.layers):
                print('training layer', i, '...')
                h_pos, h_neg = layer.train(h_pos, h_neg)

    class Layer(nn.Linear):
        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            super().__init__(in_features, out_features, bias, device, dtype)
            self.relu = torch.nn.ReLU()
            self.opt = torch.optim.Adam(self.parameters(), lr=0.03)
            self.threshold = 2.0
            self.num_epochs = 1000

        def forward(self, x):
            x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
            return self.relu(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))

        def train(self, x_pos, x_neg):
            for i in tqdm(range(self.num_epochs)):
                g_pos = self.forward(x_pos).pow(2).mean(1)
                g_neg = self.forward(x_neg).pow(2).mean(1)
                loss = torch.log(1 + torch.exp(torch.cat([
                    -g_pos + self.threshold,
                    g_neg - self.threshold]))).mean()
                lost.append(loss)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            return self.forward(x_pos).detach(), self.forward(x_neg).detach()

    torch.manual_seed(1234)
    train_loader, test_loader = train_loader, test_loader
    net = Net([X_train.shape[1], 100, 100])
    x, y = next(iter(train_loader))
    x, y = x.cuda(), y.cuda()
    x_pos = overlay_y_on_x(x, y)
    rnd = torch.randperm(x.size(0))
    x_neg = overlay_y_on_x(x, y[rnd])
    net.train(x_pos, x_neg)

    train_error = 1.0 - net.predict(x).eq(y).float().mean().item()

    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.cuda(), y_te.cuda()

    test_error = 1.0 - net.predict(x_te).eq(y_te).float().mean().item()

    return train_error, test_error, lost


# Example usage
"""

train_error, test_error, loss_values = train_and_test(
    X_train, y_train, X_test, y_test)
print("Train Error:", train_error)
print("Test Error:", test_error)
print("Loss Values:", loss_values) 
"""
