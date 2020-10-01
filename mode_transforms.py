import random
from torchvision import datasets, models, transforms,utils
import torch
import numpy as np
from PIL import Image
        
class RandomCrop(object):
    """Applies the same randomCrop to the given PIL Image and PIL segmentation target.
       Assumes both, target and Image are of the same size.
    
    """

    def __init__(self, size, mode):
        assert isinstance(size, (tuple))
        self.size = size
        self.mode = mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
 
        left = random.randint(0, h - th)
        top = random.randint(0, w - tw)
        #return left, top, right, bottom
        return left, top, left + tw, top + th


    def __call__(self, sample):
        """
        Args:
            (img,seg) where img (PIL Image): Image to be cropped.
                            seg (PIL Image): Segmentation target to be cropped.

        Returns:
            PIL Image: Cropped image and Cropped segmentation.
        """
        image, target = sample[0], sample[1]
        left, top, right, bottom = self.get_params(image, self.size)
        if self.mode == "det":
            detection = target
            return image.crop((left, top, right, bottom)), detection.crop((left, top, right, bottom))
        if self.mode == "seg":
            segmentation = target
            return image.crop((left, top, right, bottom)), segmentation.crop((left, top, right, bottom))
        if self.mode == "both":
            detection, segmentation = target[0],target[1]
            return image.crop((left, top, right, bottom)), (detection.crop((left, top, right, bottom)),segmentation.crop((left, top, right, bottom)))


def warn_input_size(input_size):
    if input_size[0] % 4 != 0:
        print("Warning: Sizes[0] are not divisible by 4.")
    if input_size[1] % 4 != 0:
        print("Warning: Sizes[1] are not divisible by 4.")

class Resize(object):
    """Resizes the image in a sample and segmentation target to a given size.
    Expects input to be PIL image.

    Args:
        output_size (tuple or int): Desired output size.
    """

    def __init__(self, output_size, mode):
        assert isinstance(output_size, (tuple))
        warn_input_size(output_size)

        self.mode = mode
        self.output_size = output_size
        self.target_size = (int(output_size[0]/4),int(output_size[1]/4))

        self.resize = transforms.Resize(self.output_size)
        self.resize_seg = transforms.Resize(self.target_size,interpolation=Image.NEAREST)
        self.resize_dec = transforms.Resize(self.target_size)

        self.sizes = []

    def __call__(self, sample):
        self.sizes.append(sample[0].size)
        if self.mode == "det":
            image, detection = sample[0], sample[1]
            return (self.resize(image), self.resize_dec(detection))
        if self.mode == "seg":
            image, segmentation = sample[0], sample[1]
            return (self.resize(image), self.resize_seg(segmentation))
        if self.mode == "both":
            image, detection, segmentation = sample[0], sample[1][0], sample[1][1]
            return self.resize(image), (self.resize_dec(detection),self.resize_seg(segmentation))

class ToTensor(object):
    """Resizes the image in a sample and segmentation target to a given size.
    Expects input to be PIL image.

    Args:
        output_size (tuple or int): Desired output size.
    """

    def __init__(self, mode):
        self.mode = mode
        self.transform = transforms.ToTensor()

    def __call__(self, sample):
        if self.mode == "det":
            image, detection = sample[0], sample[1]
            return (self.transform(image), self.transform(detection))
        if self.mode == "seg":
            image, segmentation = sample[0], sample[1]
            return (self.transform(image), torch.from_numpy(np.array(segmentation)))
        if self.mode == "both":
            image, detection, segmentation = sample[0], sample[1][0], sample[1][1]
            return self.transform(image), (self.transform(detection),torch.from_numpy(np.array(segmentation)))

class ToDevice(object):
    """ Takes Tensor and converts device and float.

    Args:
    """

    def __init__(self, mode, device):
        self.mode = mode
        self.device = device
        self.transform = lambda input_tensor: input_tensor.to(device).float()

    def __call__(self, sample):
        if self.mode == "det":
            image, detection = sample[0], sample[1]
            return self.transform(image), self.transform(detection)
        if self.mode == "seg":
            image, segmentation = sample[0], sample[1]
            return self.transform(image), self.transform(segmentation)
        if self.mode == "both":
            image, detection, segmentation = sample[0], sample[1][0], sample[1][1]
            return self.transform(image), (self.transform(detection),self.transform(segmentation))

class Normalize(object):
    """Normalizes the PIL image in a sample.

    Args:
        Normalization Parameters.
    """

    def __init__(self,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]):
        self.transform = transforms.Normalize(mean=mean, std=std)

    def __call__(self, sample):
        image, target = sample[0], sample[1]
        return self.transform(image), target
