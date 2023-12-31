import numpy as np
import nnfs 
from nnfs.datasets import spiral_data
nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    # Forward pass:
    def forward(self, inputs):
        self.inputs = inputs    #remember inputs (recall when calculating the partical derivative w.r.t. weights during backward)
        self.output = np.dot(inputs, self.weights) + self.biases
    # Backward pass:
    def backward(self, dvalues):    
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)  #gradient of the neuron function w.r.t. the weights
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)  #gradient of the neruron function w.r.t. inputs

class Activation_ReLU:
    # Forward pass:
    def forward(self, inputs):
        self.inputs = inputs     #remember inputs
        self.output = np.maximum(0, inputs)
    # Backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()   #ensure not to modify dvalues during ReLU derivative calculation
        #zero gardient where input values: negative
        self.dinputs[self.inputs <=0] = 0

class Activation_Softmax:
    # Forward pass:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
    # Backward pass
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)   #uninitialized array - same shape with dvalues array
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1,1)     # Flatten output array
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Probabilities for target values 
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    # Backward
    def backward(self, dvalues, y_true):
        samples = len(dvalues)  #number of samples
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculate gradient
        self.dinputs = -y_true / dvalues
        #Normalize gradient
        self.dinputs = self.dinputs / samples

# Combine Softmax & Cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossEntropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()
    # Forward 
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    # Backward
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        # If labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        #print('dinputs before normalize: ',self.dinputs[:10])
        #Normalize gradient
        self.dinputs = self.dinputs / samples
        #print('dinputs after normalize: ',self.dinputs[:10])

#Create dataset
X, y = spiral_data(samples=100, classes=3)
#print('Shape of X: ',X.shape)
#print('Shape of y: ',y.shape)

# 1st Dense Layer
dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()
dense1.forward(X)
activation1.forward(dense1.output)  #output of 1st Dense Layer

# 2nd Dense Layer
dense2 = Layer_Dense(3,3)
dense2.forward(activation1.output)

# Loss function
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()
loss = loss_activation.forward(dense2.output, y)
print('output of forward Loss_activation: ', loss_activation.output[:10])

# Calculate accuracy from output of activation 2 and targets
predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions==y)
#print('len of dvalues: ', len(loss_activation.output))
#print('Output of loss activation 2:', loss_activation.output[:10])
#print('prediction: ', predictions[:10])
#print('accuracy: ', accuracy)

# BAckward
loss_activation.backward(loss_activation.output, y)
#print('Shape of der w.r.t nputs of loss_act: ', loss_activation.dinputs.shape)
print('derivatives w.r.t inputs of loss_activation: ', loss_activation.dinputs[:10])
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

print(dense1.dweights)
print(dense1.dbiases)
#print(dense2.dweights)
#print(dense2.dbiases)