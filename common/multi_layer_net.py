# multi_layer_net.py
# general class definition for multi-layer neural networks
# 
import sys, os
sys.path.append(os.pardir) 
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient


class MultiLayerNet:
    """ fully connected conventional back-propagation neural networks
    Parameters
    ----------
    input_size : dimension of input （e.g. 784 in MNIST）
    hidden_size_list : number of neurons in hidden layers （e.g. [100, 100, 100]）
    output_size : dimension of output （e.g. 10 in MNIST）
    activation : 'relu' or 'sigmoid'
    weight_init_std : SD of initial weight（e.g. 0.01）
        'relu' or 'he': using the He's method for network initialization
        'sigmoid' or 'xavier': useint the Xavier's method for network initialization 
        
    weight_decay_lambda : coefficients for weight Decay（L2-norm） [regularizer]
    """
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        # weight initialization
        self.__init_weight(weight_init_std)

        # layer definition
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
            self.params['b' + str(idx)])

        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """ setting of initial weights of networks
        Parameters
        ----------
        wweight_init_std : SD of initial weight（e.g. 0.01）
            'relu' or 'he': using the He's method for network initialization
            'sigmoid' or 'xavier': useint the Xavier's method for network initialization 
        """
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # recommended initial values when using ReLU
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  # recommended initial values when using sigmoid

            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """calculate loss 
        Parameters
        ----------
        x : input data
        t : training label
        Returns
        -------
        value of the loss
        """
        y = self.predict(x)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """calculate gradient (in numererical way: too slow)
        Parameters
        ----------
        x : input
        t : training label
        Returns
        -------
        Dictionary variables with gradients for each layer
            grads['W1']、grads['W2']、... weights
            grads['b1']、grads['b2']、... bias 
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        """calculate gradient (back propagation)
        Parameters
        ----------
        x : input
        t : training label
        Returns
        -------
        Dictionary variables with gradients for each layer
            grads['W1']、grads['W2']、... weights
            grads['b1']、grads['b2']、... bias 
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # internal process ..
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.layers['Affine' + str(idx)].W
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads