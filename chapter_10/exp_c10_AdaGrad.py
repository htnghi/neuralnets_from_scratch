class Optimizer_Adagrad:
     # Initialize optimizer - set settings, learning rate of 1 is default dor this optimizer
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    #call once before any parameter updates
    def pre_update_params(self):
        #if decay rate other than 0, update current_learning_rate
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations)) 

    #update parameters
    def update_params(self,layer):
        # if layer does not contain momentum arrays, create them filled with 0
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer_weights)
            layer.bias_momentums = np.zeros_like(layer.biases)