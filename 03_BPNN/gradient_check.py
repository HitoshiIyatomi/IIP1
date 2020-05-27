# gradient_check.py
# confirm back-prop gradient is working correctly
#
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) #include parent directly
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net2 import TwoLayerNet2

# data load
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet2(input_size=784, hidden_size=20, output_size=10)

x_batch = x_train[:3] # (3,784) 3 data x 784dim
t_batch = t_train[:3] # (3, 10) 3 data x  10dim (one-hot)

#calculate gradient [dW1][dW2][db1][db2] in numerical way
grad_numerical = network.numerical_gradient(x_batch, t_batch)
#calculate gradient [dW1][dW2][db1][db2] with back-prop
grad_backprop = network.gradient(x_batch, t_batch)

# show difference of gradients  
for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))