import cv2
import numpy as np
from utils import *
from tqdm import tqdm
from loss import *
from dataloader import TestLoader, SaliconDataset
from scipy.stats import multivariate_normal
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import argparse
import glob
import os
import torch
import sys
import time
import torch.nn as nn
import pickle
from torch.distributions.multivariate_normal import MultivariateNormal as Norm
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')

parser = argparse.ArgumentParser()

parser.add_argument('--val_img_dir', default="../images/", type=str)
parser.add_argument('--model_val_path',
                    default="../saved_models/salicon_pnas.pt", type=str)
parser.add_argument('--no_workers', default=4, type=int)
parser.add_argument('--enc_model', default="pnas", type=str)
parser.add_argument('--results_dir', default="../results/", type=str)
parser.add_argument('--heatmap_dir', default="../heatmap_results/", type=str)
parser.add_argument('--validate', default=0, type=int)
parser.add_argument('--save_results', default=1, type=int)
parser.add_argument(
    '--dataset_dir', default="/home/samyak/old_saliency/saliency/SALICON_NEW/", type=str)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if args.enc_model == "pnas":
    print("PNAS Model")
    from model import PNASModel
    model = PNASModel()

elif args.enc_model == "densenet":
    print("DenseNet Model")
    from model import DenseModel
    model = DenseModel()

elif args.enc_model == "resnet":
    print("ResNet Model")
    from model import ResNetModel
    model = ResNetModel()

elif args.enc_model == "vgg":
    print("VGG Model")
    from model import VGGModel
    model = VGGModel()

elif args.enc_model == "mobilenet":
    print("Mobile NetV2")
    from model import MobileNetV2
    model = MobileNetV2()

if args.enc_model != "mobilenet":
    model = nn.DataParallel(model)
model.load_state_dict(torch.load(args.model_val_path))

model = model.to(device)

val_img_ids = os.listdir(args.val_img_dir)
val_dataset = TestLoader(args.val_img_dir, val_img_ids)
vis_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)


def validate(model, loader, device, args):
    model.eval()
    tic = time.time()
    total_loss = 0.0
    cc_loss = AverageMeter()
    kldiv_loss = AverageMeter()
    nss_loss = AverageMeter()
    sim_loss = AverageMeter()

    for (img, gt, fixations) in tqdm(loader):
        img = img.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)

        pred_map = model(img)

        # Blurring
        blur_map = pred_map.cpu().squeeze(0).clone().numpy()
        blur_map = blur(blur_map).unsqueeze(0).to(device)

        cc_loss.update(cc(blur_map, gt))
        kldiv_loss.update(kldiv(blur_map, gt))
        nss_loss.update(nss(blur_map, gt))
        sim_loss.update(similarity(blur_map, gt))

    print('CC : {:.5f}, KLDIV : {:.5f}, NSS : {:.5f}, SIM : {:.5f}  time:{:3f} minutes'.format(
        cc_loss.avg, kldiv_loss.avg, nss_loss.avg, sim_loss.avg, (time.time()-tic)/60))
    sys.stdout.flush()

    return cc_loss.avg


if args.validate:
    val_img_dir = args.dataset_dir + "images/val/"
    val_gt_dir = args.dataset_dir + "maps/val/"
    val_fix_dir = args.dataset_dir + "fixations/fixations/"

    val_img_ids = [nm.split(".")[0] for nm in os.listdir(val_img_dir)]
    val_dataset = SaliconDataset(
        val_img_dir, val_gt_dir, val_fix_dir, val_img_ids)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)
    with torch.no_grad():
        validate(model, val_loader, device, args)
if args.save_results:
    visualize_model(model, vis_loader, device, args)
