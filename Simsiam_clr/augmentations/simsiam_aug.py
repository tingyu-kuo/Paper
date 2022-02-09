import torchvision.transforms as T
import imgaug.augmenters as iaa
from PIL import Image
import numpy as np
try:
    from torchvision.transforms import GaussianBlur
except ImportError:
    from .gaussian_blur import GaussianBlur
    T.GaussianBlur = GaussianBlur
    
imagenet_mean_std = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]

class SimSiamTransform():
    def __init__(self, image_size, mean_std=imagenet_mean_std):
        image_size = 224 if image_size is None else image_size # by default simsiam use image size 224
        p_blur = 0.5 if image_size > 32 else 0 # exclude cifar

        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=p_blur),
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])
        self.jigsaw = iaa.Sequential([
            iaa.Jigsaw(nb_rows=(2, 5), nb_cols=(2, 5))
        ])
        
        self.transform_j = T.Compose([
            T.Resize([image_size, image_size]),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=p_blur),
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])
    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        # # jigsaw
        # x = [np.array(x, dtype=np.uint8)]
        # x3 = self.jigsaw.augment_images(x)
        # x3 = Image.fromarray(x3[0])
        # x3 = self.transform_j(x3)
        return x1, x2
