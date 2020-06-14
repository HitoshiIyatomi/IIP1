# image_recognition_VGG.py
# 
import sys,os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from base_transform import BaseTransform
from setup import load_VGG_pretrainednet, load_ILSVRClist
from predictor import ILSVRCPredictor
#
# args[1]: file name 
# args[2]: color normalization: True/False
#
# (ex) python image_recognition_VGG.py ./data/jpg True/False
#

# input test image to recognize
args=sys.argv
image_file_path = args[1]
print('input file is', image_file_path)
colornorm=args[2]
print('color normalization is', colornorm)

if not os.path.exists(image_file_path):
    print("file not found", image_file_path)
    print('Usage: (ex) python image_recognition_VGG.py ./data/jpg True')
    sys.exit(1)

# image open 
img = Image.open(image_file_path)  # [Height][Width][Color(RGB)]
plt.figure()
plt.imshow(img)
plt.title('input image')
plt.show(block=False)

## pre-trained network file load from local
net=load_VGG_pretrainednet()
net.eval()  #use network as inference mnode
#print(net) #display network architecture

#load ILSVRC class index
ILSVRC_class_index =load_ILSVRClist()

# transform image and format to pyTorch tensor
resize = 224
mean = (0.485, 0.456, 0.406) #mean of ImageNet dataset
std = (0.229, 0.224, 0.225)  #SD of ImageNet dataset

predictor = ILSVRCPredictor(ILSVRC_class_index)
transform = BaseTransform(resize, mean, std, colornorm)  # make a transform class
img_transformed = transform(img)  # torch.Size([3, 224, 224])

#display transformed image
img_transformed_np = img_transformed.numpy().transpose((1, 2, 0)) #numpy.ndarray([224,224,3])
img_transformed_np = np.clip(img_transformed_np, 0, 1) 
plt.figure()
plt.imshow(img_transformed_np)
plt.title('transformed image')
plt.show(block=False)

# recognition
inputs = img_transformed.unsqueeze_(0)  # torch.Size([1, 3, 224, 224])
out = net(inputs)  # torch.Size([1, 1000])

#predictor.predict_top5_show(out)
top5id, top5p=predictor.predict_top5(out)
input("Press [enter] to continue.")