import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_nerurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_nerurons)
        self.biases = np.zeros((1, n_nerurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# Softmax Activation---------------------------
class Activation_Softmax:
    def forward(self, inputs):
        # Unnormalized probabilities:
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

#Create dataset
X, y = spiral_data(samples=100, classes=3)

# 1st Dense Layer
dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()
dense1.forward(X)
activation1.forward(dense1.output)  #output of 1st Dense Layer

# 2nd Dense Layer
dense2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()
dense2.forward(activation1.output)
activation2.forward(dense2.output)  #output of 2nd Dense Layer

print(activation2.output)






