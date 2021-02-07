"""
Main program
"""
import random
import numpy as np
from neural_network import NeuralNetwork, Layer


def generate_samples(size, mean_x, mean_y, standard_deviation_x, standard_deviation_y):
    return np.random.default_rng().normal((mean_x, mean_y), (standard_deviation_x, standard_deviation_y), (size, 2))

# Globals
x_train = np.arange(400).reshape((200, 2))
y_train = np.array([1 if x < 100 else 0 for x in range(200)])
population = 20
mutation_chance = 2


def main():
    generation = 0
    best_accuracy = 0.0
    networks = [NeuralNetwork() for _ in range(population)]
    optimal_weights = []

    while best_accuracy < 0.9 and generation < 100:
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
        print(networks[0].layers[0].weights)
        for nn in networks:
            if nn.accuracy > best_accuracy:
                best_accuracy = nn.accuracy
                print('Best Accuracy: ', best_accuracy)
                for layer in nn.layers:
                    optimal_weights.append(layer.weights)
                print(optimal_weights)

        new_generation = []
        for i in range(5):
            new_generation.append(networks[i])
            for j in range(3):
                nn = networks[i]
                random_cross = random.randint(0, 5)
                for idx, layer in enumerate(nn.layers):
                    locus = random.randint(0, 1)
                    layer.weights[locus] = networks[random_cross].layers[idx].weights[locus]
                    if random.randint(0, 100) <= mutation_chance:
                        print("MUTATION!")
                        #layer.weights[0] = np.negative(layer.weights[0])
                        layer.weights[locus] = np.random.randn(np.size(layer.weights[locus]))
                new_generation.append(nn)
        networks = new_generation

    print("Over 0.90 accuracy")
    print(best_accuracy)


if __name__ == '__main__':
    main()
