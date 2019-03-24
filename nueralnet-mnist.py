import sys, os
import numpy as np
import pickle
from PIL import Image
sys.path.append("../deep-learning-from-scratch")
from dataset.mnist import load_mnist

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp = np.sum(exp_x)
    return exp_x / sum_exp


def get_data():
    (x_train, t_train), (x_test, t_test) =\
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("../deep-learning-from-scratch/ch03/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y  = softmax(a3)

    return y

input, answer = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(input)):
    y = predict(network, input[i])
    p = np.argmax(y)

    if p == answer[i]:
        accuracy_cnt += 1

print("Accuracy: " + str(float(accuracy_cnt) / len(input)))
