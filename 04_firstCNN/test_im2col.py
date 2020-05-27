# test_im2col.py 
# Checking data dimensional transformation for convolutional processing
#
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from common.util import im2col

Nb=10
C=3
Hin=20
Win=20
L=Nb*C*Hin*Win+1

x=np.arange(1,L,dtype='int').reshape(Nb,C,Hin,Win)
print(x.shape)
#print(x)

filter_h=5
filter_w=5
stride=2
padding=1

x_col=im2col(x,filter_h,filter_w,stride,padding)
print(x_col.shape)
#print(x_col)
#print("hit return")
