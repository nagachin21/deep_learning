import sys, os
#sys.path.append(os.pardir)
sys.path.append("../../deep-learning-from-scratch/")
import numpy as np
import matplotlib.pylab as plt
from dataset.mnist import load_mnist
from common.trainer import Trainer
from deep_conv_net import DeepConvNet

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = DeepConvNet()
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=100, optimizer='Adam',
                  optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)

trainer.train()

# パラメータの保存
network.save_params("deep_convnet_params.pkl")
print("Save Network Parameters!")
