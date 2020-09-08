#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
import random
import time
import utils
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


def get_files_train_val(dir,val_part=0.1, _ext='.png', strip_ext=True):
    image_dirs_assoc = {}
    for root, dirs, files in os.walk(dir):
        for fname in files:
            if fname.endswith(_ext):
                image_dirs_assoc[root]=1
                break
    image_dirs = list( image_dirs_assoc.keys() )
    random.shuffle(image_dirs)
    val_cnt = int(len(image_dirs)*val_part)

    val_dirs = image_dirs[:val_cnt]
    train_dirs=image_dirs[val_cnt:]
    # {'val': , 'train': }
    val_files=[]
    train_files=[]

    for dir in val_dirs:
        for fname in os.listdir(dir):
            if fname.endswith(_ext):
                fff=os.path.join(dir, fname)
                if strip_ext:
                    fff=os.path.splitext(os.path.join(dir,fname))[0]
                val_files.append(fff)
    for dir in train_dirs:
            for fname in os.listdir(dir):
                if fname.endswith(_ext):
                    fff = os.path.join(dir, fname)
                    if strip_ext:
                        fff = os.path.splitext(os.path.join(dir, fname))[0]
                    train_files.append(fff)

    return {'val':val_files, 'train':train_files}

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

def feed_image_content(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""
    # for id, pos in ids:
    for id, rotation in ids:

        try:
            # im1 = Image.open(dir + id + suffix)
            dbg = 1
            loaded=False
            if suffix=='.gif':
                if not os.path.isfile(id+suffix):
                    im1=Image.new('I', (512, 512),1)
                    loaded=True
            dbg = 2
            if not loaded:
                im1 = Image.open(id + suffix)
            dbg = 3
            if rotation!=0:
                im1=im1.rotate(rotation)
            # im1.show()
            # time.sleep(1)
            dbg = 4
            im = resize_and_crop(im1, scale=scale)

            # dim = np.zeros((512//4, 512//4))
            if suffix=='.jpg' or suffix=='.png': #DR fix to rbg
                #pass
                dbg = 5
                im =  np.stack((im, im, im), axis=2)
            # yield get_square(im, pos)
            yield get_square(im, 0)
        except Exception as e:
            print(dbg)
            print(id)
            print(str(e))

def get_imgs_and_masks(ids, dir_img, dir_mask, scale):
    """Return all the couples (img, mask)"""

    # imgs = to_cropped_imgs(ids, dir_img, '.jpg', scale)
    # imgs = to_cropped_imgs(ids, dir_img, '.jpg', scale)
    imgs = feed_image_content(ids, dir_img, '.png', scale)

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)

    def normalize2(x):
        return (x/8-4096)/4096

    imgs_normalized = map(normalize2, imgs_switched)

    # masks = to_cropped_imgs(ids, dir_mask, '_mask.gif', scale)

    #masks = to_cropped_imgs(ids, dir_mask, '_mask.gif', scale)
    masks = feed_image_content(ids, '', '.gif', scale)

    return zip(imgs_normalized, masks)


def get_full_img_and_mask(id, dir_img, dir_mask):
    # im = Image.open(dir_img + id + '.jpg')
    im = Image.open(dir_img + id + '.jpg')
    # mask = Image.open(dir_mask + id + '_mask.gif')
    mask = Image.open(dir_mask + id + '_mask.gif')
    return np.array(im), np.array(mask)
