# Chapter 2 - Coding our first neurons

<!-- Overview -->
## Overview

A single neuron:
* Suppose we have an input: `inputs = [1, 2, 3]`
* Weights and biases: change during the training
  
  `weights = [0.2, 0.8, -0.5]`

  `bias = 2`  # only one bias value per neuron

==> Output Calculation (for individual neuron): (inputs*weights)+bias
  
  `output = (inputs[0] * weights[0] +
             inputs[1] * weights[1] +
             inputs[2] * weights[2] + bias)`
  
  `print(output)`  # result: 2.3

A Layer of Neurons:
* Neural networks typically have layers that consist of more than one neuron.
* Layers are groups of neurons. Each neuron in a layer takes exactly the same input but contains its own set of weights and its own bias, producing its own unique output.
* Suppose we have 3 neurons in a layer and 4 inputs:

  `inputs ​= ​[​1​, ​2​, ​3​, ​2.5​]`

  `weights1 ​= ​[​0.2​, ​0.8​, ​-​0.5​, ​1​]` `weights2 ​= ​[​0.5​, ​-​0.91​, ​0.26​, ​-​0.5​]` `weights3 ​= ​[​-​0.26​, ​-​0.27​, ​0.17​, ​0.87​]`

  `bias1 ​= ​2` `bias2 ​= ​3`  `bias3 ​= ​0.5`

  `outputs ​= ​[

          ​# Neuron 1:
          ​inputs[​0​]​*​weights1[​0​] ​+ 
          ​inputs[​1​]​*​weights1[​1​] ​+ 
          ​inputs[​2​]​*​weights1[​2​] ​+ 
          ​inputs[​3​]​*​weights1[​3​] ​+ ​bias1,
          ​# Neuron 2: 
          ​inputs[​0​]​*​weights2[​0​] ​+ 
          ​inputs[​1​]​*​weights2[​1​] ​+ ​
          inputs[​2​]​*​weights2[​2​] ​+ ​
          inputs[​3​]​*​weights2[​3​] ​+ ​bias2,
          ​# Neuron 3: ​
          inputs[​0​]​*​weights3[​0​] ​+ 
          ​inputs[​1​]​*​weights3[​1​] ​+ ​
          inputs[​2​]​*​weights3[​2​] ​+ 
          ​inputs[​3​]​*​weights3[​3​] ​+ ​bias3]`

* Use a loop: 
* ==> This is called a ​fully connected​ neural network — every neuron in the current layer has connections to every neuron from the previous layer.

<!-- Examples -->

