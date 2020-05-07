#while true; do nvidia-smi ; sleep 1; done
#while true; do  banner `nvidia-smi |head -9|tail -1| cut -d\  -f 5` ; sleep 1; done
import datetime
import sys
import os
from optparse import OptionParser
import numpy as np
import utils.data_vis as data_vis

import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from eval import eval_net
from unet import UNet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch, print2
from PIL import Image
from globals import RUN_NAME
N_CHANNELS=3


# import utils
def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.05,
              save_cp=True,
              gpu=False,
              img_scale=0.5):
    # ddd = "/media/d/old data 3TB/GitHub/"
    # dir_img = ddd+'train/'
    # dir_mask = ddd+'train_masks/'


    current_lr = lr

    ddd = "/home/d/Pytorch-UNet/"
    dir_img = ddd+'perc_train/'
    dir_mask = ddd+'perc_train_mask/'
    dir_checkpoint = 'checkpoints/'

    ddd = "/home/d/_Denis/"
    dir_img = ddd + 'train/'
    dir_mask = ddd + 'train_mask/'
    dir_checkpoint = f'checkpoints{RUN_NAME}/'

    dir_img = '/home/d/train_data_polygon/'
    # dir_img = '/home/d/Pytorch-UNet/train_test_data'
    # dir_img = '/media/d/ssd256 win10 2018/_Denis/MarkSet1_augment'
    ids = get_ids(dir_img)

    #ids = split_ids(ids,1)## strange
    ids = split_ids(ids,1) ## strange


    arr_epoch_loss=[]
    arr_val_dice = []
    iddataset = split_train_val(ids, val_percent)

    print2('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(iddataset['train']),
               len(iddataset['val']), str(save_cp), str(gpu)))

    N_train = len(iddataset['train'])

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    criterion = nn.BCELoss()
    time_start = time.time()
    for epoch in range(epochs):
        time_start_epoch = time.time()
        print2(datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S"))
        print2('Starting epoch {}/{} lr={}.'.format(epoch + 1, epochs,current_lr))

        optimizer = optim.SGD(net.parameters(),
                              lr=current_lr,
                              momentum=0.9,
                              weight_decay=0.0005)
        current_lr=0.8*current_lr


        net.train()

        # reset the generators
        train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_scale)
        val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale)

        epoch_loss = 0

        for i, b in enumerate(batch(train, batch_size)):
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1] for i in b])

            imgs = torch.from_numpy(imgs)


            true_masks = torch.from_numpy(true_masks)

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()


            masks_pred = net(imgs)
            masks_probs_flat = masks_pred.view(-1)

            true_masks_flat = true_masks.view(-1)

            if i<100:
                last_masks_pred = masks_pred[0][0].data.cpu().numpy()
                last_img = imgs[0][0].data.cpu().numpy()
                last_true_masks = true_masks[0].data.cpu().numpy()

                last_masks_pred*=255
                img_last_masks_pred = Image.fromarray(last_masks_pred)
                img_last_img = Image.fromarray(last_img)

                last_true_masks*=255
                img_last_true_masks = Image.fromarray(last_true_masks)
                data_vis.plot_img_and_mask(img_last_img, img_last_masks_pred, img_last_true_masks, f'ep{epoch}_i{i:03}.jpg')

            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()

            print2('{0} {1:.4f}% --- loss: {2:.6f}'.format(epoch,  100*i * batch_size / N_train, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # break
        # if i:
        arr_epoch_loss.append(epoch_loss / i)
        for l in arr_epoch_loss:
            print2('Epoch finished ! Loss: {}'.format(l))

        if 1:
            time_start_validation = time.time()
            val_dice = eval_net(net, val, gpu, epoch,iddataset['val'])
            arr_val_dice.append(val_dice)
            for d in arr_val_dice:
                print2('Validation Dice Coeff: {}'.format(d))
        if 1:
            time_now = time.time()
            diff1 = int(time_now - time_start_epoch)
            diff2 = int(time_now - time_start_validation)
            diff3 = int(time_now - time_start)
            print2(f'epoch={diff1} seconds, validation={diff2} seconds, total={diff3} seconds')

        if save_cp:
            os.makedirs(dir_checkpoint, exist_ok=True)
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print2('Checkpoint {} saved !'.format(epoch + 1))



def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=10,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()
    args.gpu=True
    args.scale=0.50
    args.lr=0.01
    args.batchsize=12
    args.epochs=20
    # args.batchsize=1
    # args.load='INTERRUPTED.pth'

    # print('writting to out.txt')
    # orig_stdout = sys.stdout
    # f = open('out.txt', 'w', buffering=100)
    # sys.stdout = f




    net = UNet(n_channels=N_CHANNELS, n_classes=1)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print2('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale,
                  val_percent=0.1
                  )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print2('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
