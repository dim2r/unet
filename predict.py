import argparse
import datetime
import os
import pydicom as dicom
import numpy as np
import torch
import torch.nn.functional as F
import json

from PIL import Image

from unet import UNet
from utils import resize_and_crop, normalize, split_img_into_squares, hwc_to_chw, merge_masks, dense_crf
from utils import plot_img_and_mask
import math

from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


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
        # output_right = net(X_right)

        left_probs = output_left.squeeze(0)
        # right_probs = output_right.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(img_height),
                transforms.ToTensor()
            ]
        )
        
        left_probs = tf(left_probs.cpu())
        # right_probs = tf(right_probs.cpu())

        left_mask_np = left_probs.squeeze().cpu().numpy()
        # right_mask_np = right_probs.squeeze().cpu().numpy()

    # full_mask = merge_masks(left_mask_np, right_mask_np, img_width)
    full_mask=left_mask_np
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
    start_datetime_string = datetime.datetime.today().strftime("%Y_%m_%d__%H_%M")
    head_is_printed = False
    # args = get_args()
    # f='/home/d/train_data_polygon/MarkSet1_original/set1/abdomen/1/1.2.392.200036.9116.2.6.1.48.1214242851.1571977503.267444'
    f='/home/d/train_data_polygon/MarkSet1_original/set1/chest/1-50/49/1.2.392.200036.9116.2.6.1.48.1214242851.1571826533.534693'
    #2 spots
    # f='/home/d/train_data_polygon/MarkSet1_original/set1/patalogy/calcium/49/1.2.392.200036.9116.2.6.1.48.1214242851.1571915481.25008'
    # in_files = args.input
    # in_masks = args.xxx
    # out_files = get_output_filenames(args)

    in_files = [f+'.png']
    in_masks = [f+'.gif']
    out_files = ['out.gif'] #get_output_filenames(args)
    in_model = '/home/d/Pytorch-UNet/checkpoints3/CP20.pth'

    rootdir = '/media/';
    # rootdir = '/home/d/test_data';

    net = UNet(n_channels=3, n_classes=1)

    print(f"Loading model {in_model}...")


    if torch.cuda.is_available():
        print("Using CUDA !")
        net.cuda()
        net.load_state_dict(torch.load(in_model))
        use_cuda = True
    else:
        net.cpu()
        net.load_state_dict(torch.load(in_model, map_location='cpu'))
        use_cuda = False
        print("Using CPU, this may be very slow.....")

    print("Model loaded")
    i=0
    for currentDir, subdirs, files in os.walk(rootdir):
        dicom_files = []
        for f in files:
            if f.find('.dcm') > -1:
                dicom_files.append(f)
        dicom_files.sort()
        count = len(dicom_files)
        i=0
        for f in dicom_files:
            if f.find('.dcm') > -1:
                i+=1
                fn = currentDir + '/' + f
                ds = dicom.dcmread(fn)
                pixel_array_numpy = ds.pixel_array
                pixel_array_numpy += 2048
                pixel_array_numpy <<= 3
                pixel_array_numpy = pixel_array_numpy.astype(np.uint32)

                # img = Image.open(fn)
                img = Image.fromarray(pixel_array_numpy)
                if img.size[0] < img.size[1]:
                    print("Error: image height larger than the width")

                head_str = 'calc;N;NN;n_clusters;size1;size2;dir;file'
                info_file_name =  fn.replace('.dcm','.json')
                info_file_exists = False
                if os.path.isfile(info_file_name):
                    info_file_exists=True



                def print_string(print_str, currentDir):
                    print(print_str)
                    path_parts = currentDir.split("/", 4)
                    path_to_write = '/'.join(path_parts[:4])
                    fname=f'{path_to_write}/DICOM_ANALYSIS_{start_datetime_string}.csv'
                    if os.path.isfile(fname):
                        head_is_printed = True
                    else:
                        head_is_printed = False

                    with open(fname, "a") as myfile:
                        if not head_is_printed:
                            head_is_printed = True
                            myfile.write(f'{head_str}\n')

                        myfile.write(f'{print_str}\n')
                if info_file_exists:
                    with open(info_file_name,'r') as fhandle:
                        info_data = json.load(fhandle)

                        # info_data = {'N': i, 'NN': count, "n_clusters": n_clusters_, 'size1': sz1, 'size2': sz2,
                        #              'dir': currentDir, 'file': f}
                        i=info_data['N']
                        count=info_data['NN']
                        n_clusters_=info_data['n_clusters']
                        sz1=info_data['size1']
                        sz2=info_data['size2']

                        print_str = f'load;{i};{count};{n_clusters_};{sz1};{sz2};{currentDir};{f}'
                        print_string(print_str, currentDir)

                if not info_file_exists:
                    mask = predict_img(net=net,
                                       full_img=img,
                                       scale_factor=0.5,
                                       out_threshold=0.5,
                                       use_dense_crf= False,
                                       use_gpu=use_cuda)

                    have_an_object=False
                    scatter_data=[]
                    for x in range(len(mask)):
                        for y in range(len(mask[x])):
                            if mask[x][y]==False:
                                scatter_data.append([x,y])

                    if len(scatter_data)==0:
                        info_data = {'N': i, 'NN': count, "n_clusters":0, 'size1': 0, 'size2': 0,
                                     'dir': currentDir, 'file': f}
                        with open(info_file_name, 'w') as fhandle:
                            json.dump(info_data, fhandle)

                    if len(scatter_data):
                        try:
                            pca = PCA(2)
                            pca.fit(scatter_data)
                            transformed_normalized = pca.transform(scatter_data)
                            min_x = None
                            max__X = None
                            min_y = None
                            max__Y = None
                            for k in range(len(transformed_normalized)):
                                x=transformed_normalized[k][0]
                                y=transformed_normalized[k][1]
                                if min_x is None or min_x>x:
                                    min_x=x
                                if min_y is None or min_y>y:
                                    min_y=y
                                if max__X is None or max__X<x:
                                    max__X=x
                                if max__Y is None or max__Y<y:
                                    max__Y=y
                            sz1=max__X - min_x
                            sz2=max__Y - min_y

                            db = DBSCAN(eps=2, min_samples=5).fit(scatter_data)
                            labels = db.labels_
                            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                            sz1=int(round(sz1))
                            sz2=int(round(sz2))
                            have_an_object=True


                            info_data = {'N':i, 'NN':count, "n_clusters" : n_clusters_, 'size1': sz1, 'size2': sz2, 'dir':currentDir,'file':f}
                            with open(info_file_name,'w') as fhandle:
                                json.dump(info_data, fhandle)
                            print_str=f'calc;{i};{count};{n_clusters_};{sz1};{sz2};{currentDir};{f}'
                            if i==1:
                                print(head_str)

                            print_string(print_str,currentDir)

                            result = mask_to_image(mask)
                            out_fn = fn.replace('.dcm', '.gif')
                            result.save(out_fn)

                        except Exception as e:
                            print(str(e))

