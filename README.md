# Genetic algorithm optimization of start weights in                       a neural network

This program is an university project, which focus on combination of the genetic algorithms and neural networks. The program realized idea of finding very good weights and biases for neural network, and start teaching NN with them. Thanks to usage of GA in purpose of finding good initial weights for NN, time and number of epochs in which NN will achieve classification accuracy over 90% is smaller.

Diagram of genetic algorithm created for the purpose of this project:

<img src="gen_algo_diagram.png" alt="algo diagram" style="zoom:80%;" />

## Program structure

Program struct is very simple, in repository there are two python files. In NeuralNetwork.py can be find class which represent NN, Layer class which have all weights, biases and activation function from one NN layer. Custom NN are built from 3 Layers, especially for the research case. Also in this file there are a few types of activation function (ReLu, Heavy, Sigmoid). In the main.py there is whole code for genetic algorithm. The code is describe in python help and with some comments for genetic parts.

## How to run

Best way is to open repository folder as PyCharm project, this give an easy way to download all needed modules. After that, create new run configuration by adding python interpreter and main.py as main function. Last, start the program.

To start program from command line, first needed is to download all requirements and then type: python main.py  

## Use genetic with Keras

For this, code need some rework in genetic part.  First of all, it is needed to create keras model and then by model method which give weights from layer, replace all manually weights swapping in genetic. 

## Requirements

Python 3.7.9
Numpy 1.19.2
Pandas 1.1.3
Python math module