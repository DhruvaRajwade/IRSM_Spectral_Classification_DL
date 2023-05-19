#Just discovered PINNS! This is a simple implementation of a PINN for the 2D Poisson equation
#I'm using the PyTorch framework for this implementation
# I'm in love with PINNS, this is a great option to do a PhD on!


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define the PDE


def f(x, y):
    return torch.sin(x) + torch.cos(y)

# Define the PINN


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.dense1 = nn.Linear(2, 32)
        self.dense2 = nn.Linear(32, 32)
        self.dense3 = nn.Linear(32, 1)

    def forward(self, inputs):
        x, y = inputs[:, 0:1], inputs[:, 1:2]
        u = torch.tanh(self.dense1(torch.cat([x, y], 1)))
        u = torch.tanh(self.dense2(u))
        u = self.dense3(u)
        return u

# Define the loss function


def loss(model, x, y):
    inputs = torch.cat([x, y], 1)
    u = model(inputs)
    u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(
        u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    f_pred = u_x + u_y - f(x, y)
    mse = torch.mean(torch.square(f_pred))
    return mse


# Create the PINN model
model = PINN()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the PINN
for epoch in range(100):
    x = torch.rand([100, 1], requires_grad=True) * 2 * 3.1416
    y = torch.rand([100, 1], requires_grad=True) * 2 * 3.1416
    loss_value = loss(model, x, y)

    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()

    if epoch % 1 == 0:
        print("Epoch {}: Loss = {}".format(epoch, loss_value.item()))

# Visualize the outputs
x_vals = torch.linspace(0, 2 * 3.1416, 100)
y_vals = torch.linspace(0, 2 * 3.1416, 100)
X, Y = torch.meshgrid(x_vals, y_vals)
inputs = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], 1)
u_pred = model(inputs)
U = u_pred.detach().numpy().reshape(100, 100)

plt.figure()
plt.contourf(X, Y, U, levels=20, cmap='viridis')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('PINN Solution')
plt.show()
