import numpy as np
from abc import ABC, abstractmethod


class Layer(ABC):
    """Base Class for Layers
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
        """
        Initializes layer's matrix W and b

        Arguments:
            input_shape {int} -- [previous layer shape]
        """
        self.W = np.zeros((self.size, prev_layer_size))
        self.b = np.zeros((self.size, 1))

    @abstractmethod
    def forward_activation(self, A_prev):
        """
        Performs forward activation over the layer based on input

        Arguments:
            A_prev { np_ matrix } -- [Previous layer's output]

        Returns:
            [np_matrix] -- [Layer's output after activation]
        """
        pass

    def update_gradients(self, dW, db):
        """Sets or updates layer's gradients

        Arguments:
            dW { np_matrix } -- [New value for dW]
            db { np_matrix } -- [New value for db]
        """
        self.dW = dW
        self.db = db

    @abstractmethod
    def update_parameters(self, learning_rate):
        """
            Updates Layer parameters

        Arguments:
            learning_rate { float } -- [Learning rate value]
        """
        pass


class Dense(Layer):
    """Regular densely-connected NN layer.
    layer_output = activation(dot(input, weights) + bias)
    Note: The input to the layer should be flattened if has a rank greater
    than 2.

    # Arguments
        activation: Activation function to use.
        size: layer size
    """

    def build(self, input_shape):
        """
        Initializes layer's matrix W and b

        Arguments:
            input_shape {[int]} -- [previous layer shape]
        """
        self.__create_W(input_shape)
        self.__create_b()

    def __create_W(self, prev_layer_size):
        self.W = np.random.randn(
            self.size,
            prev_layer_size
        ) * 0.1

    def __create_b(self):
        self.b = np.zeros((self.size, 1))

    def forward_activation(self, A_prev):
        """
        Performs forward activation over the layer based on input

        Arguments:
            A_prev { np_matrix } -- [Previous layer's output]

        Returns:
            A { np_matrix } -- [Layer's output after activation]
        """
        # Z = W*X + b
        # A = g(Z)
        self.Z = np.add(np.dot(self.W, A_prev), self.b)
        self.A = self.activation(self.Z)
        return self.A

    def update_parameters(self, learning_rate):
        """
            Updates Layer's parameters

        Arguments:
            learning_rate { float } -- [Learning rate value]
        """
        self.W = self.W - learning_rate * self.dW
        self.b = self.b - learning_rate * self.db
