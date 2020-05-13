import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from unet import UNet
from utils import resize_and_crop, normalize, split_img_into_squares, hwc_to_chw, merge_masks, dense_crf
from utils import plot_img_and_mask

from torchvision import transforms

def predict_img(net,
                full_img,
                scale_factor=0.5,
                out_threshold=0.5,
                use_dense_crf=False,
                use_gpu=True):

    net.eval()
    img_height = full_img.size[1]
    img_width = full_img.size[0]

    img = resize_and_crop(full_img, scale=scale_factor)
    img = normalize(img)
    img = np.stack((img, img, img), axis=2)
    left_square, right_square = split_img_into_squares(img)

    left_square = hwc_to_chw(left_square)
    right_square = hwc_to_chw(right_square)

    X_left = torch.from_numpy(left_square).unsqueeze(0)
    X_right = torch.from_numpy(right_square).unsqueeze(0)
    
    if use_gpu:
        X_left = X_left.cuda()
        X_right = X_right.cuda()

    with torch.no_grad():
        output_left = net(X_left)
        output_right = net(X_right)

        left_probs = output_left.squeeze(0)
        right_probs = output_right.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(img_height),
                transforms.ToTensor()
            ]
        )
        
        left_probs = tf(left_probs.cpu())
        right_probs = tf(right_probs.cpu())

        left_mask_np = left_probs.squeeze().cpu().numpy()
        right_mask_np = right_probs.squeeze().cpu().numpy()

    full_mask = merge_masks(left_mask_np, right_mask_np, img_width)

    if use_dense_crf:
        full_mask = dense_crf(np.array(full_img).astype(np.uint8), full_mask)

    return full_mask > out_threshold



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")

    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--xxx', '-x', metavar='INPUT', nargs='+',
                        help='filenames of mask images' )

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='filenames of ouput images')

    parser.add_argument('--cpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=False)
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--no-crf', '-r', action='store_true',
                        help="Do not use dense CRF postprocessing",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=1)

    return parser.parse_args()

def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

if __name__ == "__main__":
    # args = get_args()
    f='/home/d/train_data_polygon/MarkSet1_original/set1/abdomen/1/1.2.392.200036.9116.2.6.1.48.1214242851.1571977503.267444'
    # in_files = args.input
    # in_masks = args.xxx
    # out_files = get_output_filenames(args)

    in_files = [f+'.png']
    in_masks = [f+'.gif']
    out_files = ['out.gif'] #get_output_filenames(args)
    in_model = '/home/d/Pytorch-UNet/checkpoints3/CP20.pth'
    net = UNet(n_channels=3, n_classes=1)

    print("Loading model {}".format(in_model))

    if torch.cuda.is_available():
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
        net.load_state_dict(torch.load(in_model))
    else:
        net.cpu()
        net.load_state_dict(torch.load(in_model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    print("Model loaded !")

    for i, fn in enumerate(in_files):
        print("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)


        if img.size[0] < img.size[1]:
            print("Error: image height larger than the width")

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=0.5,
                           out_threshold=0.5,
                           use_dense_crf= False,
                           use_gpu=True)


        if False:
            #python3 predict.py --model checkpoints/CP7.pth -i perc_train/aaghewgysa.jpg -o 1.jpg --xxx perc_train_mask/aaghewgysa_mask.gif -v
            if len(in_masks) > 0:
                img_mask = Image.open(in_masks[0])

            print("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask, img_mask,'../predict_plot.jpg')

        if True:
            out_fn = out_files[i]

            min_x_pos=10000
            max_x_pos=-10000
            for x in range(len(mask)):
                cnt=0
                for y in range(len(mask[x])):
                    if mask[x][y]==False:
                        cnt+=1
                if cnt>0 and x<min_x_pos:
                    min_x_pos=x
                if cnt>0 and x>max_x_pos:
                    max_x_pos=x
            size=max_x_pos-min_x_pos
            print(f'size={size} at min_x_pos={min_x_pos} ... max_x_pos={max_x_pos}')
            result = mask_to_image(mask)
            result.save(out_files[i])

            print("Mask saved to {}".format(out_files[i]))
