import gc
import torch
import torch.nn.functional as F
import utils.data_vis as data_vis
from dice_loss import dice_coeff
from PIL import Image

def eval_net(net, dataset, gpu=False,epoch=None, ids=None):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    filename=''
    for i, b in enumerate(dataset):
        img = b[0]
        true_mask = b[1]
        if ids:
            filename=ids[i][0]
            filename=filename.replace('/','_')
            filename=filename.replace('\\','_')
            filename=filename.replace(':','_')

        img = torch.from_numpy(img).unsqueeze(0)
        true_mask = torch.from_numpy(true_mask).unsqueeze(0)

        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()

        mask_pred = net(img)[0]
        if True or i<500:
            last_masks_pred = mask_pred[0].data.cpu().numpy()
            last_masks_pred *= 255
            last_img = img[0][0].data.cpu().numpy()
            last_true_masks = true_mask[0].data.cpu().numpy()
            last_true_masks *= 255
            img_last_masks_pred = Image.fromarray(last_masks_pred)

            min_ = last_img.min()
            max_ = last_img.max()
            img_last_img = Image.fromarray(255 * (last_img - min_) / (max_ - min_))

            img_last_true_masks = Image.fromarray(last_true_masks)


            data_vis.plot_img_and_mask(img_last_img, img_last_masks_pred, img_last_true_masks, f'{filename}_ep{epoch}_i{i:03}_eval.jpg')
            gc.collect()
        if i % 300 == 0:
            print(i)

        mask_pred = (mask_pred > 0.5).float()

        tot += dice_coeff(mask_pred, true_mask).item()



    return tot / (i + 1)
