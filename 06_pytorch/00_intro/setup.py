# setup.py
# preliminary setup
# 1) Load ILSVRC class list (1000 class)
# 2) Load VGG-pretrained model
# If you don't have them locally, download them.
#
import os
import urllib.request
import zipfile
import pickle
import json
from torchvision import models

data_dir = "./data/"

def load_ILSVRClist():
    #  if data folder does not exist, create it
    #data_dir = "./data/"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    #download ILSVRC class index and save it as "./data/imagenet_class_index.json"
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    save_path = os.path.join(data_dir, "imagenet_class_index.json")

    if not os.path.exists(save_path):
        print('loading ILSVRC class index from web')
        urllib.request.urlretrieve(url, save_path)
    
    print('read ILSVRC class index')
    index_list = json.load(open('./data/imagenet_class_index.json', 'r'))
    return index_list


def load_VGG_pretrainednet():
    net_file = data_dir + "VGG_pretrained_net.pkl"
    # if not exists, download (from torchvision) and save it 
    # (only at the first time) 
    if not os.path.exists(net_file):
        print('VGG network file not found - start loading from web ')
        use_pretrained = True
        net = models.vgg16(pretrained=use_pretrained)
        print('VGG network file load from web - done')
        print('save network file on',net_file)
        f=open(net_file,'wb')
        pickle.dump(net,f)
        f.close
        print('save network file [done]')

    else:
        with open(net_file,'rb') as f:
            net=pickle.load(f)
            print('loaded VGG network from local')
    return net
