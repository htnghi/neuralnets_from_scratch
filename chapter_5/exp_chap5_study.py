import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

# Classification: dog (class 0 - index 0), cat (class 1 - index 1), human (class 2 - index 2)
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                   [0.1, 0.5, 0.4],
                   [0.02, 0.9, 0.08]])
class_targets = [0, 1, 1]   # doa, cat, cat

# Get list of the confidences at the target indices
#print(softmax_outputs[range(len(softmax_outputs)), class_targets])

# Negative Log Calculation
#print(-np.log(softmax_outputs[range(len(softmax_outputs)), class_targets]))

neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
average_loss = np.mean(neg_log) # The average loss per batch
print(average_loss)

