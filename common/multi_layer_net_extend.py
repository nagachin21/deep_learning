import sys, os
sys.path.append(os.pardir)
import numpy as np
from collections import OrderdDict
from common.layers import *
from common.gradient import numerical_gradient

class MultiLayerNetExtend:
    """全結合による多層ニューラルネットワークの拡張版

    weight Decay, Dropout,  Batch Normalizationの機能を持つ

    Parameters
    -----------
    input_size: 入力データのサイズ(MNISTの場合は784)
    hidden_size_list: 隠れ層のニューロン数のリスト(e.g. [100, 100, 100])
    output_size: 出力データのサイズ(MNISTの場合は10)
    activation: 'relu' または 'sigmoid'
    weight_init_std: 重みを初期化する際の標準偏差の設定(e.g. 0.01)
        'relu' または 'He' を指定した場合は「Heの初期値」
        'sigmoid' または 'Xavier' を指定した場合は「Xavierの初期値」
    weight_decay_lambda: Weight Decay（L2ノルム）の強さ
    use_dropout: Dropoutを使用するかどうか
    dropout_ration: Dropoutの割合
    use_batchnorm: Batch Normalizationを使用するかどうか
    """
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0,
                 use_dropout=False, dropout_ration=0.5, use_batchnorm=False):
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.output_size = output_size
        self.hidden_layer_num = len(hidden_size_list)
        self.use_dropout = use_dropout
        self.weight_decay_lambda = weight_decay_lambda
        self.use_batchnorm = use_batchnorm
        self.params = {}

        #重みの初期化
        self.__init_weight(weight_init_std)

        #レイヤの生成
        activation_layer = {'sigmoid':Sigmoid, 'relu':Relu}
        self.layers = OrderdDict()
        for idx in range(1, hidden_layer_num + 1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])

            if use_batchnorm:
                self.params['gamma' + str(idx)] = np.ones(hidden_size_list[idx - 1])
                self.params['beta' + str(idx)] = np.zeros(hidden_size_list[idx - 1])
                self.layers['BatchNorm' + str(idx)] = BatchNormalization(self.params['gamma' + str(idx)],
                                                                         self.params['beta' + str(idx)])
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()

            if use_dropout:
                self.layers['Dropout' + str(idx)] = Dropout(dropout_ration)

        idx = hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """重みの初期化

        Parameters
        ----------
        weight_init_std: 重みを初期化する際の標準偏差(e.g. 0.01)
            'relu' または 'He' を指定した場合は「Heの初期値」
            'sigmoid' または 'Xavier' を指定した場合は「Xavierの初期値」
        """
        all_size_list = [self.input_size] + hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower in ('relu', 'He'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1]) # Reluを利用する際に推奨される初期値
            elif str(weight_init_std).lower in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1]) # Sigmoidを利用する際に推奨される初期値
            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx - 1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])
