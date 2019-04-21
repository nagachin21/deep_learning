import sys
sys.path.append("../deep-learning-from-scratch/")
import numpy as np
from dataset.mnist import load_mnist
from revised_two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num  = 10000
train_size = x_train.shape[0]
batch_size = 10000
learning_rate = 0.1

train_loss_list = []
train_acc_list  = []
test_acc_list   = []

iters_per_epoch
