# Chapter 1 -  Introducing Neural Networks

* “Artificial” neural networks are inspired by the organic brain, translated to the computer.
* A single neuron by itself is relatively useless, but, when combined with hundreds or thousands (or many more) of other neurons, the interconnectivity produces relationships and results that frequently outperform any other machine learning methods.


# Chapter 2 - Coding our first neurons
<!-- Overview -->
## Overview

A single neuron:
* Suppose we have inputs to this neuron
* Parameters: at the beginning, weights initialized randomly and biases set as zero '0'
    - Note: Weights and biases will change inside the model during the training phase.
* The neuron's output value: `output = inputs * weights + bias`

A Layer of Neurons:
* Neural networks typically have layers that consist of more than one neuron.
* Layers are groups of neurons. Each neuron in a layer takes exactly the same input but contains its own set of weights and its own bias, producing its own unique output.
* The ​fully connected​ neural network: every neuron in the current layer has connections to every neuron from the previous layer

==> A single neuron and a layer of neurons with Numpy
`outputs = np.dot(weights, inputs) + biases`

A batch of Data
* NNs tend to receive data in ​batches to train
* NNs expect to take in many ​samples​ (also called as feature set instances or observations) at a time for two reasons. One reason is that it’s faster to train in batches in parallel processing, and the other reason is that batches help with generalization during training.
==> `layer_ouput = np.dot(inputs, np.array(weights).T) + biases`

# Chapter 3 -  Adding layers

* Neural networks become “deep” when they have 2 or more ​hidden layers​ (which layers between these endpoints have values that we don’t necessarily deal with)
* Note: in following chapters, we will use nnfs.dataset to generate data (which will be explicit in example codes)

Dense Layer (also called as fully-connected layer)
* Forward method: When we pass data through a model from beginning to end, it is called a forward pass.
* Note: 

# Chapter 4 -  Activation Functions
* The activation function is applied to the output of a neuron (or layer of neurons), which modifies outputs. 
* The reason why using the activation functions because it itself is nonlinear, it allows for neural networks with usually two or more hidden layers to map nonlinear functions.
* 2 types: used in hidden layers and used in output layers.
* Some options for activation functions:
- Sigmoid Activation Function: `σ(z) = 1 / (1 + e^(-z))`
Outputs values between 0 and 1 -> suitable for binary classification problems where you want to model probabilities.
- Rectified Linear Unit (ReLU) Activation Function: `ReLU(z) = max(0, z)`
Outputs the input for positive values and zero for negative values.
The most widely used activation function at the time of writing for various reasons — mainly speed and efficiency.
- The Softmax Activation Function: (Output layer for classification)
This activation can take non-normalized, or uncalibrated, inputs and produce a normalized distribution of probabilities for each classes. This distribution returned by the softmax activation function represents ​confidence scores​ for each class and will add up to 1. The predicted class is associated with the output neuron that returned the largest confidence score.

# Chapter 5 -  Calculating Network Error with Loss

* Loss function (also referred to as the cost function): is the algorithm that quantifies how wrong a model is.
* Loss​ is the measure of this metric. Since loss is the model’s error, we ideally want it to be 0.



# Chapter 14 - L1 and L2 Regularization
* L1 and L2 Regularization: Add penalty number to the loss function to penalize the model for large w and b.\
*Note: large w indicate that a neuron memorizes a data element*
    - L1 (rarely used alone) encourages sparsity in weights, effectively setting some weights to exactly zero => help with feature selection and simplifying the model
    - L2 (commonly used): encourages the model to have smaller weights overall => effectively discourages the model from relying too heavily on any single weight => reducing the risk of overfitting.




<!-- Examples -->

