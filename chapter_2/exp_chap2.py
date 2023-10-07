inputs = [1.0, 2.0, 3.0, 4.0]
weights = [[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

import numpy as np
outputs = np.dot(weights, inputs) + biases

print('Outputs exp 2: ', outputs)