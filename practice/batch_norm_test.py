import sys, os
sys.path.append(os.pardir)
sys.path.append("../../deep-learning-from-scratch/")
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist impor load_mnist
from multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD, ADAM
