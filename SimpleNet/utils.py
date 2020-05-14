import cv2
import os
import torch
from os.path import join
import matplotlib.pyplot as plt
from torchvision import transforms, utils
import numpy as np
from tqdm import tqdm
from PIL import Image


def blur(img):
    k_size = 11
    bl = cv2.GaussianBlur(img, (k_size, k_size), 0)
    return torch.FloatTensor(bl)


def plot(pred, gt, args, idx):
    pred_npimg = utils.make_grid(pred.cpu()).numpy()
    gt_npimg = utils.make_grid(gt.cpu()).numpy()

    plt.figure(figsize=(10, 10))
    plt.subplot(121)
    plt.title("Original")
    plt.imshow(np.transpose(gt_npimg, (1, 2, 0)))

    plt.subplot(122)
    plt.title("Predicted")
    plt.imshow(np.transpose(pred_npimg, (1, 2, 0)))

    plt.savefig(args.results_dir + '{}_{}.png'.format(epoch, idx+1))


def visualize_model(model, loader, device, args):
    with torch.no_grad():
        model.eval()
        os.makedirs(args.results_dir, exist_ok=True)

        for (img, img_id, sz) in tqdm(loader):
            img = img.to(device)

            pred_map = model(img)
            pred_map = pred_map.cpu().squeeze(0).numpy()
            pred_map = cv2.resize(pred_map, (sz[0], sz[1]))

            pred_map = torch.FloatTensor(blur(pred_map))
            img_save(pred_map, join(
                args.results_dir, img_id[0]), normalize=True)

            heat_map = im2heat(join(args.results_dir, img_id[0]), join(
                args.val_img_dir, img_id[0]))

            im = Image.fromarray(heat_map)
            exten = join(args.heatmap_dir, img_id[0]).split('.')[-1]
            if exten == "png":
                im.save(join(args.heatmap_dir,
                             img_id[0]), format=format, compress_level=0)
            else:
                # for jpg
                im.save(join(args.heatmap_dir,
                             img_id[0]), format=format, quality=100)


def img_save(tensor, fp, nrow=8, padding=2,
             normalize=False, range=None, scale_each=False, pad_value=0, format=None):
    grid = utils.make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                           normalize=normalize, range=range, scale_each=scale_each)

    ''' Add 0.5 after unnormalizing to [0, 255] to round to nearest integer '''

    ndarr = torch.round(grid.mul(255).add_(0.5).clamp_(
        0, 255).permute(1, 2, 0)).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    exten = fp.split('.')[-1]
    if exten == "png":
        im.save(fp, format=format, compress_level=0)
    else:
        im.save(fp, format=format, quality=100)  # for jpg


class AverageMeter(object):

    '''Computers and stores the average and current value'''

    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count


def convert(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def im2heat(map_path, rgb_image_path):
    gt = cv2.imread(rgb_image_path)
    gt = convert(gt)
    pred = cv2.imread(map_path, 0)
    heatmap_img = cv2.applyColorMap(pred, cv2.COLORMAP_JET)
    heatmap_img = convert(heatmap_img)
    pred = np.stack((pred, pred, pred), 2).astype('float32')
    pred = pred / 255.0
    return np.uint8(pred * heatmap_img + (1.0-pred) * gt)
