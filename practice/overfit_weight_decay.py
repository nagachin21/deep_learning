import numpy as np
import sys
sys.path.append("../")
sys.path.append("../../deep-learning-from-scratch/")
import matplotlib.pylab as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

#過学習を再現するために、学習データを削減
x_train = x_tarin[:300]
t_train = t_train[:300]

# weight decay（荷重減衰）の設定   =====================
#weight_decay_lambda = 0 # weight decayを使用しない場合
#weight_decay_lambda = 0.1
#====================================================

network
