# base_transform.py
# This class provides color normalization
#  provided in Residual network (He, 2015) to improve performance
import torchvision
from torchvision import models, transforms

class BaseTransform():
    """
    resize image into (resize x resize)
    normalize color into N(mean, std)

    Attributes
    ----------
    resize : int
        size of the image to be transformed
    mean : (R, G, B)
        mean color value to be transformed
    std : (R, G, B)
        standard deviation of color value to be transformed
    colornorm: [True] / False
        flag of color normalization
    """
    

    def __init__(self, resize, mean, std, colornorm=True):
        if colornorm=='True':
            self.base_transform = transforms.Compose([
                transforms.Resize(resize),  # length of minor axis =resize
                transforms.CenterCrop(resize),  # center crop with resize x resize
                transforms.ToTensor(),  # convert into Torch tensor
                transforms.Normalize(mean, std)  # normalize color
            ])
        else:
          self.base_transform = transforms.Compose([
                transforms.Resize(resize),  # length of minor axis =resize
                transforms.CenterCrop(resize),  # center crop with resize x resize
                transforms.ToTensor(),  # convert into Torch tensor
                #transforms.Normalize(mean, std)  # normalize color
            ])  


    def __call__(self, img):
        return self.base_transform(img) 
 