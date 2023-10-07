import numpy as np
import nnfs 
from nnfs.datasets import spiral_data
nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# loss class
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)
        print('Len of sample (y_pred): ', len(y_pred))
        print('Shape of sample (y_pred): ', y_pred.shape) 
        print('Range of sample (y_pred): ', range(samples))  
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        print('Shape y prediction clipped:', y_pred_clipped.shape)
        print('y prediction clipped:', y_pred_clipped[98:110])

        # Probabilities for target values:
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
            print('Shapes of Correct conffidences: ', correct_confidences.shape)
            print('Correct conffidences: ', correct_confidences[98:110])
            print('y true:', y_true[98:110])
            
        #Mask values - only for one-hot encoded labels:
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        print('Neg_log: ', negative_log_likelihoods)
        return negative_log_likelihoods


#Create dataset
X, y = spiral_data(samples=100, classes=3)
print('Shape of X: ',X.shape)
print('Shape of y: ',y.shape)
print('y label:', y[108:110])

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

# Loss function
loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)   #take out put of 2nd Dense layer here and return output

print('loss', loss)
