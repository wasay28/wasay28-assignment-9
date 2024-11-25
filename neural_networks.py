import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function
        # TODO: define layers and initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))
        # Initialize storage dictionaries
        self.activations = {'hidden': None, 'output': None}
        self.gradients = {'W1': None, 'W2': None}
        
    def forward(self, X):
        # TODO: forward pass, apply layers to input X
        self.z1 = np.dot(X, self.W1) + self.b1
        if self.activation_fn == 'tanh':
            self.a1 = np.tanh(self.z1)
        elif self.activation_fn == 'relu':
            self.a1 = np.maximum(0, self.z1)
        else:  # sigmoid
            self.a1 = 1 / (1 + np.exp(-self.z1))
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        # TODO: store activations for visualization
        out = np.tanh(self.z2)
        self.activations['hidden'] = self.a1
        self.activations['output'] = out
        return out

    def backward(self, X, y):
        # TODO: compute gradients using chain rule
        m = X.shape[0]

        delta2 = (self.activations['output'] - y) * (1 - self.activations['output']**2)
        dW2 = np.dot(self.a1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        if self.activation_fn == 'tanh':
            da1 = 1 - self.a1**2
        elif self.activation_fn == 'relu':
            da1 = (self.z1 > 0).astype(float)
        else:  # sigmoid
            da1 = self.a1 * (1 - self.a1)

        delta1 = np.dot(delta2, self.W2.T) * da1
        dW1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0, keepdims=True)

        # TODO: update weights with gradient descent

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2


        # TODO: store gradients for visualization

        self.gradients['W1'] = dW1
        self.gradients['W2'] = dW2

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)
        
    # TODO: Plot hidden features
    hidden_features = mlp.activations['hidden']
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7)

    # TODO: Hyperplane visualization in the hidden space
    xx, yy = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
    zz = -(mlp.W2[0] * xx + mlp.W2[1] * yy + mlp.b2[0]) / (mlp.W2[2] + 1e-6)
    ax_hidden.plot_surface(xx, yy, zz, alpha=0.2)

    # TODO: Distorted input space transformed by the hidden layer
    xx, yy = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = mlp.forward(grid_points).reshape(xx.shape)
    ax_input.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.5)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr')

    # TODO: Plot input layer decision boundary
    layer_pos = {'input': [-2, 0], 'hidden': [0, 0], 'output': [2, 0]}

    for i, layer in enumerate(['input', 'hidden', 'output']):
        num_nodes = 2 if layer == 'input' else 3 if layer == 'hidden' else 1
        for j in range(num_nodes):
            y_pos = (j - (num_nodes-1)/2) * 0.5
            circle = Circle([layer_pos[layer][0], y_pos], 0.1, fill=False)
            ax_gradient.add_patch(circle)

    # TODO: Visualize features and gradients as circles and edges 
    # The edge thickness visually represents the magnitude of the gradient
    max_thickness = 3
    for i in range(2):  # Input layer
        for j in range(3):  # Hidden layer
            grad = np.abs(mlp.gradients['W1'][i,j])
            thickness = grad / (np.max(np.abs(mlp.gradients['W1'])) + 1e-6) * max_thickness
            ax_gradient.plot([layer_pos['input'][0], layer_pos['hidden'][0]],
                           [i*0.5 - 0.25, j*0.5 - 0.5],
                           'k-', linewidth=thickness)
    
    for i in range(3):  # Hidden to output connections
        grad = np.abs(mlp.gradients['W2'][i,0])
        thickness = grad / (np.max(np.abs(mlp.gradients['W2'])) + 1e-6) * max_thickness
        ax_gradient.plot([layer_pos['hidden'][0], layer_pos['output'][0]],
                        [i*0.5 - 0.5, 0],
                        'k-', linewidth=thickness)
    
    ax_gradient.set_xlim(-3, 3)
    ax_gradient.set_ylim(-2, 2)
    ax_gradient.axis('equal')


def visualize(activation, lr, step_num, output_file):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')  
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, 
                       ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save with the provided filename
    ani.save(output_file, writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num, "results/visualize_test.gif")