from torchvision import transforms
from PIL import Image
from torchvision.transforms import InterpolationMode

imagenet_norm = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]

class Transform_single():
    def __init__(self, image_size, train, normalize=imagenet_norm):
        if train == True:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, interpolation=InterpolationMode.BICUBIC),  # crop first and then resize
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*normalize)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(int(image_size*(8/7)), interpolation=InterpolationMode.BICUBIC), # 224 -> 256 
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(*normalize)
            ])

    def __call__(self, x):
        return self.transform(x)
