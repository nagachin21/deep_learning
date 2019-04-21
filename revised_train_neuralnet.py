import sys
sys.path.append("../deep-learning-from-scratch/")
import numpy as np
import matplotlib.pylab as plt
from dataset.mnist import load_mnist
from NN.revised_two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num  = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list  = []
test_acc_list   = []

iters_per_epoch = max(iters_num / batch_size, 1)


for i in range(iters_num):
    batch_mask = np.random.choice(iters_num, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #誤差逆伝播法によって重みの値を更新する
    grads = network.gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grads[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iters_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc  = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("iteration: " + str(i))
        print(train_acc, test_acc)

acc_iteration = np.arange(0, iters_num, batch_size)
fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4), sharex=True)

axL.plot(acc_iteration, test_acc_list, linewidth=2)
axL.set_title('test accuracy')
axL.set_xlabel('iteration')
axL.set_ylabel('accuracy')
axL.grid(True)

iteration = np.arange(0, iters_num, 1)

axR.plot(iteration, train_loss_list, linewidth=2)
axR.set_title('training loss')
axR.set_xlabel('iteration')
axR.set_ylabel('train loss')
axR.grid(True)

fig.show()
