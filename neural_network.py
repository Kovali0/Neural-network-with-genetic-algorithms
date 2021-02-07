"""
File with all code for neural network
"""
import math
import numpy as np


def ReLu(x):
    return np.maximum(0, x)


def ReLu_Derivative(x):
    return np.where(x <= 0, 0, 1)


class Layer:
    def __init__(self, size, activation_function, input_size):
        self.size = size
        self.activation_fun = activation_function
        self.input_size = input_size
        self.weights_number = size * input_size
        self.weights = np.random.randn(size, input_size)
        self.biases = np.random.randn(size, 1)


class NeuralNetwork:
    def __init__(self):
        self.l1 = Layer(2, 0, 2)
        self.l2 = Layer(3, 0, self.l1.size)
        self.l3 = Layer(2, 0, self.l2.size)
        self.layers = []
        self.layers.append(self.l1)
        self.layers.append(self.l2)
        self.layers.append(self.l3)
        self.layers_number = 3
        self.accuracy = 0

    def train(self, samples, labels, epochs, test_x=None, test_y=None, batch_size=1, learning_rate=0.1, decay=0.1):
        n = batch_size
        batches = [(samples[n * i: n * (i + 1)], labels[n * i: n * (i + 1)]) for i in range(math.ceil(len(samples) / n))]
        for e in range(epochs):
            print("epoch_{}: {}%".format(e, self.accuracy))
            current_learning_rate = learning_rate / (1 + e * decay)
            for X, y in batches:
                self._train_batch(X, y, current_learning_rate)

    def step_forward(self, samples, layer):
        activation_output = layer.weights.dot(samples) + layer.biases
        return ReLu(activation_output)

    def predict(self, samples):
        prediction = samples
        for layer in self.layers:
            prediction = self.step_forward(prediction, layer)
        return tuple(prediction)
