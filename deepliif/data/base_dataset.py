"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, dataroot):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.root = dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(preprocess, load_size, crop_size, size):
    w, h = size
    new_h = h
    new_w = w
    if preprocess == 'resize_and_crop':
        new_h = new_w = load_size
    elif preprocess == 'scale_width_and_crop':
        new_w = load_size
        new_h = load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - crop_size))
    y = random.randint(0, np.maximum(0, new_h - crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(preprocess, load_size, crop_size, no_flip, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    preprocess = preprocess or []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))  # added by Zeeon.N:  params- num_output_channel
    if 'resize' in preprocess:
        osize = [load_size, load_size]
        transform_list.append(transforms.Resize(osize, method)) 
    elif 'scale_width' in preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, load_size, crop_size, method)))
        # added by Zeeon.N:  torchvision.transforms.Lambda: Apply a user-defined lambda as a transform.
    if 'crop' in preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], crop_size)))

    if preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    """
    added by Zeeon:
    if h, w are not multiple of 4, print warning info and adjusted the img to the nearest multiple of 4 numbers.
    """
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_size, crop_size, method=Image.BICUBIC):
    """
    added by Zeeon: rescale the img while keeping the ratio , or make height equals crop_size when (target_size * oh / ow < crop_size)
    """
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:  # added by Zeeon: not sure what crop_size do here.
        return img
    w = target_size # added by Zeeon: target_size is the ideal width
    h = int(max(target_size * oh / ow, crop_size))  # added by Zeeon: keep the ratio as the original (ow, oh) ratio.
    return img.resize((w, h), method)


def __crop(img, pos, size):
    """
    added by Zeeon: a simple crop operation.
    """
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
