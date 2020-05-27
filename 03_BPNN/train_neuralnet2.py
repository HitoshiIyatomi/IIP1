# train_neuralnet2.py
#  training codes for two_layer_net2.py
######  
# copy [mnist.pkl] downloaded last time to ISP/dataset/ 
#   then you can skip download it again
#
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dataset.mnist import load_mnist
from two_layer_net2 import TwoLayerNet2

# read data
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet2(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000   # set the number of training iteration
train_size = x_train.shape[0]
batch_size = 100    # set the batch size
learning_rate = 0.1 # set the learning rate at gradient descent

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # gradient
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    # update weights
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    # display the progress of the training
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("epoch= {:5d} train_acc = {:.4f}, test_acc = {:.4f}".format(int(i/iter_per_epoch),train_acc, test_acc))