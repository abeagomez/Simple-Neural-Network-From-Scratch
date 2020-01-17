import numpy as np


class Dense():
    """Regular densely-connected NN layer.
    layer_output = activation(dot(input, weights) + bias)`
    Note: The input to the layer should be flattened if has a rank greater
    than 2.
    # Arguments
        activation: Activation function to use.
        size: layer size
    """

    def __init__(self, size, activation):
        self.size = size
        self.activation = activation
        self.W = None
        self.b = None
        self.A = None
        self.Z = None
        self.db = None
        self.dW = None
        self.name = None

    def set_name(self, index):
        self.name = "layer_" + str(index)

    def build(self, input_shape):
        self.init_W(input_shape)
        self.init_b()
        self.build = True

    def call(inputs):
        return self.forward_activation(inputs)

    def init_W(self, prev_layer_size):
        self.W = np.random.randn(
            self.size,
            prev_layer_size
        ) * 0.1

    def init_b(self):
        self.b = np.zeros((self.size, 1))

    def forward_activation(self, A_prev):
        # Z = W*X + b
        # A = g(Z)
        self.Z = np.add(np.dot(self.W, A_prev), self.b)
        self.A = self.activation(self.Z)
        return self.A

    def update_gradients(self, dW, db):
        self.dW = dW
        self.db = db

    def update_parameters(self, learning_rate):
        self.W = self.W - learning_rate * self.dW
        self.b = self.b - learning_rate * self.db
