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

    def train(self, samples, labels, epochs, batch_size=1, learning_rate=0.05, decay=0.1):
        samples, labels = self._shuffle(samples.T, labels.T)
        n = batch_size
        batches = [(samples[:, n * i: n * (i + 1)], labels[n * i: n * (i + 1)]) for i in range(math.ceil(len(samples) / n))]
        for e in range(epochs):
            self.calculate_accuracy(samples, labels)
            print("epoch_{}: {}%".format(e, self.accuracy))
            current_learning_rate = learning_rate / (1 + e * decay)
            for X, y in batches:
                next_input = X
                weighted_inputs = []
                activations = [np.copy(next_input)]
                for layer in self.layers:
                    next_input, weighted_sum = self._step_forward(next_input, layer, with_weighted_sum=True)
                    weighted_inputs.append(weighted_sum)
                    activations.append(next_input)
                dw, db = self.backpropagation(y, weighted_inputs, activations)
                for idx, layer in enumerate(self.layers):
                    layer.weights += current_learning_rate * dw[idx]
                    layer.biases += current_learning_rate * db[idx]

    def backpropagation(self, labels, weighted_inputs, activations):
        deltas = [None] * len(self.layers)
        ones = np.ones(labels.shape)
        zeros = np.zeros(labels.shape)
        y = np.where(labels > 0, [ones, zeros], [zeros, ones])
        print(y)
        print(activations[-1])
        print(activations[-1] - y)
        print(ReLu_Derivative(weighted_inputs[-1]))
        deltas[-1] = (y - activations[-1]) * ReLu_Derivative(weighted_inputs[-1])
        for i in reversed(range(len(deltas) - 1)):
            deltas[i] = self.layers[i + 1].weights.T.dot(deltas[i + 1]) * ReLu_Derivative(weighted_inputs[i])
        batch_size = labels.shape[0]
        delta_bias = [d.dot(np.ones((batch_size, 1))) / float(batch_size) for d in deltas]
        delta_weights = [d.dot(activations[i].T) / float(batch_size) for i, d in enumerate(deltas)]
        return delta_weights, delta_bias

    def _shuffle(self, samples, labels):
        indices = np.random.permutation(labels.shape[0])
        return np.array(samples[:, indices]), np.array(labels[indices])

    def _step_forward(self, samples, layer, *, with_weighted_sum=None):
        weighted_sum = layer.weights.dot(samples) + layer.biases
        if with_weighted_sum:
            return ReLu(weighted_sum), weighted_sum
        else:
            return ReLu(weighted_sum)

    def predict(self, samples):
        prediction = samples
        for layer in self.layers:
            prediction = self._step_forward(prediction, layer)
        return tuple(prediction)

    def calculate_accuracy(self, samples, labels):
        pred_1, pred_2 = self.predict(samples)
        pred = np.where(pred_1 > pred_2, 1, 0)
        results = np.where(pred == labels, 1, 0)
        self.accuracy = np.mean(results)
        return self.accuracy
