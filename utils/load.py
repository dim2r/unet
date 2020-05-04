#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
from PIL import Image

from .utils import resize_and_crop, get_square, normalize, hwc_to_chw


def get_ids_Old(dir):
    """Returns a list of the ids in the directory"""
    return (f[:-4] for f in os.listdir(dir))

def get_ids_Old_(dir):
    """Returns a list of the ids in the directory"""
    res = []
    for root, subdirs, files in os.walk(dir):
        for f in files:
            if f.find('.jpg')>-1:

                gif = f.replace('.jpg','.gif')
                if gif in files:
                    fff = f.replace('.jpg','')
                    res.append(root+'/'+fff)
    return res
    #return (f[:-4] for f in os.listdir(dir))

def get_ids(dir):
    """Returns a list of the ids in the directory"""
    res = []
    for root, subdirs, files in os.walk(dir):
        for f in files:
            if f.find('.png')>-1:
                fff = f.replace('.png','')
                res.append(root+'/'+fff)
    return res


def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, i)  for id in ids for i in range(n))


def to_cropped_imgs_old(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""
    for id, pos in ids:
        # im1 = Image.open(dir + id + suffix)
        im1 = Image.open(id + suffix)
        im = resize_and_crop(im1, scale=scale)

        # dim = np.zeros((512//4, 512//4))
        if suffix=='.jpg' or suffix=='.png': #DR fix to rbg
            #pass
            im =  np.stack((im, im, im), axis=2)
        yield get_square(im, pos)

def to_cropped_imgs(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""
    for id, pos in ids:
        # im1 = Image.open(dir + id + suffix)
        loaded=False
        if suffix=='.gif':
            if not os.path.isfile(id+suffix):
                im1=Image.new('I', (512, 512),1)
                loaded=True

        if not loaded:
            im1 = Image.open(id + suffix)

        im = resize_and_crop(im1, scale=scale)

        # dim = np.zeros((512//4, 512//4))
        if suffix=='.jpg' or suffix=='.png': #DR fix to rbg
            #pass
            im =  np.stack((im, im, im), axis=2)
        yield get_square(im, pos)

def get_imgs_and_masks(ids, dir_img, dir_mask, scale):
    """Return all the couples (img, mask)"""

    # imgs = to_cropped_imgs(ids, dir_img, '.jpg', scale)
    # imgs = to_cropped_imgs(ids, dir_img, '.jpg', scale)
    imgs = to_cropped_imgs(ids, dir_img, '.png', scale)

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    # masks = to_cropped_imgs(ids, dir_mask, '_mask.gif', scale)

    #masks = to_cropped_imgs(ids, dir_mask, '_mask.gif', scale)
    masks = to_cropped_imgs(ids, '', '.gif', scale)

    return zip(imgs_normalized, masks)


def get_full_img_and_mask(id, dir_img, dir_mask):
    # im = Image.open(dir_img + id + '.jpg')
    im = Image.open(dir_img + id + '.jpg')
    # mask = Image.open(dir_mask + id + '_mask.gif')
    mask = Image.open(dir_mask + id + '_mask.gif')
    return np.array(im), np.array(mask)
