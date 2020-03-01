import argparse
import glob, os
import torch
import sys
import time
import torch.nn as nn
import pickle
from torch.distributions.multivariate_normal import MultivariateNormal as Norm
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np, cv2
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from scipy.stats import multivariate_normal
from dataloader import TestLoader
from loss import *
from tqdm import tqdm
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--val_img_dir',default="../images/", type=str)
parser.add_argument('--model_val_path',default="../saved_models/salicon_pnas.pt", type=str)
parser.add_argument('--no_workers',default=4, type=int)
parser.add_argument('--enc_model',default="pnas", type=str)
parser.add_argument('--results_dir',default="../results/", type=str)

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

model = nn.DataParallel(model)
model = model.to(device)
model.load_state_dict(torch.load(args.model_val_path))

val_img_ids = os.listdir(args.val_img_dir)
val_dataset = TestLoader(args.val_img_dir, val_img_ids)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)

visualize_model(model, val_loader, device, args)