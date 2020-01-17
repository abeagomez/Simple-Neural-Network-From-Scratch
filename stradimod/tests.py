import unittest
import numpy as np
from layers import Dense
from utils import sigmoid, relu


class TestDenseMethods(unittest.TestCase):

    def test_set_name(self):
        layer = Dense(10, sigmoid)
        layer.set_name(1)
        self.assertEqual(layer.name, "layer_1")

    def test_w_shape_after_build(self):
        layer = Dense(10, sigmoid)
        layer.build(20)
        self.assertEqual(layer.W.shape, (10, 20))

    def test_b_shape_after_build(self):
        layer = Dense(10, sigmoid)
        layer.build(20)
        self.assertEqual(layer.b.shape, (10, 1))

    def test_layer_output_shape_after_forward_activation(self):
        layer = Dense(10, sigmoid)
        layer.build(20)
        layer.forward_activation(np.random.randn(20, 35))
        self.assertEqual(layer.A.shape, (10, 35))

    def test_update_gradient(self):
        layer = Dense(10, sigmoid)
        dW = np.random.randn(5, 10)
        db = np.random.randn(5, 1)
        layer.update_gradients(dW, db)
        self.assertEqual(layer.dW.all(), dW.all())
        self.assertEqual(layer.db.all(), db.all())

    def test_update_parameters(self):
        layer = Dense(10, sigmoid)
        layer.build(20)
        learning_rate = 0.1
        dW = np.random.randn(10, 20)
        db = np.random.randn(10, 1)
        layer.update_gradients(dW, db)
        new_dW = layer.dW - learning_rate * dW
        new_db = layer.db - learning_rate * db
        layer.update_parameters(learning_rate)
        self.assertEqual(layer.dW.all(), new_dW.all())
        self.assertEqual(layer.db.all(), new_db.all())


if __name__ == '__main__':
    unittest.main()
