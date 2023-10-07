import numpy as np
import nnfs 
from nnfs.datasets import spiral_data
nnfs.init()

class Layer_Dense:
    
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
        # forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
        #print('shape weight: ', self.weights.shape)

# ReLU activation---------------------------------------------------------
class Activation_ReLU:
    # Forward pass
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

X, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2,3)

# Create ReLU activation (use with Dense Layer)
activation1 = Activation_ReLU()

dense1.forward(X)

# Forward pass through Activation function
# Takes in output from previous layer
activation1.forward(dense1.output)
print(activation1.output)
