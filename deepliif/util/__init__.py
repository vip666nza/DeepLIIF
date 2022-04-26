"""This package includes a miscellaneous collection of useful helper functions."""
import os
import collections

import torch
import numpy as np
from PIL import Image, ImageOps

from .visualizer import Visualizer

# Postfixes not to consider for segmentation
excluding_names = ['Hema', 'DAPI', 'DAPILap2', 'Ki67', 'Seg', 'Marked', 'SegRefined', 'SegOverlaid', 'Marker', 'Lap2']
# Image extensions to consider
image_extensions = ['.png', '.jpg', '.tif']


def allowed_file(filename):
    name, extension = os.path.splitext(filename)
    image_type = name.split('_')[-1]  # Read image type

    return extension in image_extensions and image_type not in excluding_names


def chunker(iterable, size):
    for i in range(size):
        yield iterable[i::size]

# by Zeeon: 具名元组，可以理解为创建了一个名为Tile的类，其属性有i, j, img
Tile = collections.namedtuple('Tile', 'i, j, img')


def output_size(img, tile_size):
    # by Zeeon: 如果原始图像的width, height比tile_size小，就返回tile_size
    #           如果原始图像的width, height比tile_size大，就返回n*tile_size，使得结果是tile_size的倍数，且与原始w,h最近
    return (max(round(img.width / tile_size) * tile_size, tile_size),
            max(round(img.height / tile_size) * tile_size, tile_size))


def generate_tiles(img, tile_size, overlap_size):
    # by Zeeon: 从一副较大的图象上，分割出若干tile_size大小的image
    img = img.resize(output_size(img, tile_size))
    # Adding borders with size of given overlap around the whole slide image
    img = ImageOps.expand(img, border=overlap_size, fill='white')
    rows = int(img.height / tile_size)  # Number of tiles in the row
    cols = int(img.width / tile_size)  # Number of tiles in the column

    # Generating the tiles
    for i in range(cols):
        for j in range(rows):
            yield Tile(j, i, img.crop((
                i * tile_size, j * tile_size,
                i * tile_size + tile_size + 2 * overlap_size,
                j * tile_size + tile_size + 2 * overlap_size
            )))


def stitch(tiles, tile_size, overlap_size):
    # by Zeeon: 将上一步generate_tiles函数的逆操作
    rows = max(t.i for t in tiles) + 1
    cols = max(t.j for t in tiles) + 1

    width = tile_size * cols
    height = tile_size * rows

    new_im = Image.new('RGB', (width, height))

    for t in tiles:
        img = t.img.resize((tile_size + 2 * overlap_size, tile_size + 2 * overlap_size))
        img = img.crop((overlap_size, overlap_size, overlap_size + tile_size, overlap_size + tile_size))

        new_im.paste(img, (t.j * tile_size, t.i * tile_size))

    return new_im
