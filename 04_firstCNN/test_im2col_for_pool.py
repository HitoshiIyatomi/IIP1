# test_im2col_for_pool.py 
# checking the detail of internal process of max-pooling
#
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from common.util import im2col

Nb=1 # for pooling, set this 1
C=3
Hin=4
Win=4
L=Nb*C*Hin*Win+1

x=np.arange(1,L,dtype='int').reshape(Nb,C,Hin,Win)
print(x.shape)
print(x)

filter_h=2
filter_w=2
stride=2
padding=0

x_col=im2col(x,filter_h,filter_w,stride,padding)
print("x_col.shape= ", x_col.shape)
print("x_col =\n", x_col)

x_col_flat=x_col.reshape(-1, filter_h*filter_w)
print("x_col_flat.shape= ",x_col_flat.shape)
print("x_col_flat = \n",x_col_flat)

max_out = np.max(x_col_flat, axis=1)
print("max_put.shape= ",max_out.shape)
print("max_out = \n",max_out)

out_h = int(1 + (Hin - filter_h) / stride)
out_w = int(1 + (Win - filter_w) / stride)

out_reshape=max_out.reshape(Nb, out_h, out_w, C)
print("out_reshape.shape= ",out_reshape.shape)
print("out_reshape = \n",out_reshape)

out= out_reshape.transpose(0, 3, 1, 2)
print("out.shape= ",out.shape)
print("out = \n",out)

print("hit return")