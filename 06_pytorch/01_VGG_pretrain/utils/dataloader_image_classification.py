import glob
import os.path as osp
import torch.utils.data as data
from torchvision import models, transforms
from PIL import Image


class ImageTransform():
    """
    image pre-processing class. 
    This has two different modes : 'train' and 'val'
    
    'train': (data augmentation: DA) random crop, random flip
             (color normalization)
    'val'   :(center cropping)+(color normalization) 

    Attributes
    ----------
    resize : int
        size of the image to be transformed
    mean : (R, G, B)
        mean color value to be transformed
    std : (R, G, B)
        standard deviation of color value to be transformed
    """

    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(      # DA: random crop (resize, change aspect ratio 4:3 or 3:2)
                    resize, scale=(0.5, 1.0)),     #   length of minor axis = scale * resize 
                transforms.RandomHorizontalFlip(), # DA: random horizontal flip
                transforms.ToTensor(),             # convert into Torch tensor
                transforms.Normalize(mean, std)    # color normalization
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),     # length of minor axis = resize
                transforms.CenterCrop(resize), # center crop (resize)×(resize)
                transforms.ToTensor(),
                transforms.Normalize(mean, std) 
            ])
        }

    def __call__(self, img, phase='train'):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
        """
        return self.data_transform[phase](img)


def make_datapath_list(phase="train"):
    """
    Create a list containing the paths to the data

    Parameters
    ----------
    phase : 'train' or 'val'

    Returns
    -------
    path_list : list
        list containing the paths to the data
    """

    rootpath = "./data/hymenoptera_data/"
    target_path = osp.join(rootpath+phase+'/**/*.jpg')
    print(target_path)

    path_list = []  # path will be stored

    # Using glob to get the file path to a subdirectory
    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list


class HymenopteraDataset(data.Dataset):
    """
    Dataset class for ant and bee images
    inherited by Dataset class of PyTorch

    Attributes
    ----------
    file_list : list
        list containing the paths to the data
    transform : object
        instance of image transform (pre-processing)
    phase : 'train' or 'test'
        flag whether train or test
    """

    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list  # list containing the paths to the data
        self.transform = transform  # instance of image transform (pre-processing)
        self.phase = phase          # train or val

    def __len__(self):
        '''return number of images'''
        return len(self.file_list)

    def __getitem__(self, index):
        '''
        obtain data (in Tensor) and label of pre-processed images
        '''

        # load index-th image
        img_path = self.file_list[index]
        img = Image.open(img_path)  # [Height][Width][Color=RGB]

        # perform pre-process (transform) 
        img_transformed = self.transform(img, self.phase)  # torch.Size([3, 224, 224])

        # extract label info from image (directory) name
        if self.phase == "train":
            label = img_path[30:34]
        elif self.phase == "val":
            label = img_path[28:32]

        # convert label into int
        if label == "ants":
            label = 0
        elif label == "bees":
            label = 1

        return img_transformed, label