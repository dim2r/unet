import matplotlib.pyplot as plt
from numba import jit
import os
from globals import RUN_NAME
#@jit
def plot_img_and_mask(img, mask, mask2, filename ):

    fig = plt.figure()
    a = fig.add_subplot(1, 3, 1)
    a.set_title('Input image')
    plt.imshow(img)

    b = fig.add_subplot(1, 3, 2)
    b.set_title('UNet mask')
    plt.imshow(mask)

    if mask2 is not None:
        c = fig.add_subplot(1, 3, 3)
        c.set_title('True mask')
        plt.imshow(mask2)

    # fig.set_canvas(plt.gcf().canvas)

    # plt.show()
    dir = f'/home/d/Pytorch-UNet/train_plots{RUN_NAME}/'
    os.makedirs(dir, exist_ok=True)
    plt.savefig(dir+filename+'.png')
    plt.close()