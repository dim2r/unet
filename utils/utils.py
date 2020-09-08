import random
import time
import numpy as np
import os.path


import os
import sys


def get_square(img, pos):
    """Extract a left or a right square from ndarray shape : (H, W, C))"""
    h = img.shape[0]
    if pos == 0:
        return img[:, :h]
    else:
        return img[:, -h:]

def split_img_into_squares(img):
    return get_square(img, 0), get_square(img, 1)

def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

def resize_and_crop(pilimg, scale=0.5, final_height=None):

    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    img = pilimg.resize((newW, newH))
    img = img.crop((0, diff // 2, newW, newH - diff // 2))

    #DR fix rotation
    w = img.size[0]
    h = img.size[1]
    img = img.crop((int(w*0.10), int(h*0.10), int(w*0.90), int(h*0.90)))
    #END DR
    # img.show()
    # time.sleep(15)

    return np.array(img, dtype=np.float32)

def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b

def split_train_val(dataset, val_percent=0.05):
    """DR: group by patients"""
    dataset = list(dataset)

    path_assoc_list_all={}
    path_assoc_list_NO_AUG={}
    for d in dataset:
        path = os.path.dirname(d[0])
        path_assoc_list_all[path]=1+path_assoc_list_all.get(path, 0)
        if path.find('augment')==-1:
            path_assoc_list_NO_AUG[path]=1+path_assoc_list_NO_AUG.get(path, 0)

    path_list = list( path_assoc_list_all.keys() )
    path_list_NO_AUG = list( path_assoc_list_NO_AUG.keys() )
    random.shuffle(path_list)
    random.shuffle(path_list_NO_AUG)



    length = len(path_list_NO_AUG)
    n = int(length * val_percent)

    val_path_list_NO_AUG = path_list_NO_AUG[-n:]

    val_set =[]
    train_set=[]

    for id in dataset:
        to_val=False
        for val_path in val_path_list_NO_AUG:
            if id[0].find(val_path)>-1:
                to_val=True
                val_set.append(id)
                # print('train ' + id[0])
                break

        if not to_val: #to train set
            found_augment=False
            for val_path in val_path_list_NO_AUG:
                aug_path = val_path.replace('_original','_augment')
                if id[0].find(aug_path) > -1:
                    found_augment=True
            if not found_augment:
                train_set.append(id)
                # print('val '+id[0])


    random.shuffle(train_set)
    random.shuffle(val_set)
    return {'train': train_set, 'val': val_set}

    #OLD
    # length = len(dataset)
    # n = int(length * val_percent)
    # random.shuffle(dataset)
    # return {'train': dataset[:-n], 'val': dataset[-n:]}


def normalize(x):
    return x / 255

def merge_masks(img1, img2, full_w):
    h = img1.shape[0]

    new = np.zeros((h, full_w), np.float32)
    new[:, :full_w // 2 + 1] = img1[:, :full_w // 2 + 1]
    new[:, full_w // 2 + 1:] = img2[:, -(full_w // 2 - 1):]

    return new


# credits to https://stackoverflow.com/users/6076729/manuel-lagunas
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


def print2(s):
    f = open("out.txt", "a")
    f.write(s)
    f.write("\n")
    f.close()
    print(s)