"""
Main program
"""
import numpy as np
from neural_network import NeuralNetwork, Layer


# Globals
x_train = np.arange(200).reshape((100, 2))
y_train = np.array([1 if x < 50 else 0 for x in range(100)])
population = 20


def main():
    generation = 0
    best_accuracy = 0.0
    networks = [NeuralNetwork() for _ in range(population)]

    while best_accuracy < 0.9 or generation < 101:
        generation += 1
        print("========== Generation number ", generation, " ==========")

        for nn in networks:
            good_prediction = 0
            for i, x in enumerate(x_train):
                pred_1, pred_2 = nn.predict(x)
                if pred_1 > pred_2 and y_train[i] == 1 or pred_1 < pred_2 and y_train[i] == 0:
                    good_prediction += 1
            nn.accuracy = good_prediction / np.size(x_train)

        networks = sorted(networks, key=lambda z: z.accuracy)
        networks.reverse()
        for nn in networks:
            if nn.accuracy > best_accuracy:
                best_accuracy = nn.accuracy
                print('Best Accuracy: ', best_accuracy)
                optimal_weights = []
                for layer in nn.layers:
                    optimal_weights.append(layer.weights)
                print(optimal_weights)

        for i in range(4):
          pass

if __name__ == '__main__':
    main()
