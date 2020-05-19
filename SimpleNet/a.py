import torch
from torchvision import transforms, utils
from torch.utils.data import DataLoader
from dataloader import TestLoader

from loss import *
from utils import *
from model import PNASModel
from model import MobileNetV2
import time
from PIL import Image
import logging

class SaliencyInference:
    def __init__(self):
        self.model_path = '../saved_models/mobilenet.pt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MobileNetV2()
        self.model.load_state_dict(torch.load(self.model_path), strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.image_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def get_saliency_map(self, image, t):
        """
        1. resize image
        2. convert to tensor
        3. normalize
        """
        sz = image.size
        img = self.image_transform(image)
        # print("t image transform", time.time() - t)
        img = img.unsqueeze(0)

        with torch.no_grad():
            img = img.to(self.device)
            # print(6, time.time()-t)
            #print("type: {}, size: {}, shape: {}".format(type(img), sz, img.size()))
            pred_map = self.model(img)
            # print(7, time.time()-t)
            pred_map = pred_map.cpu().squeeze(0).numpy()
            # print(8, time.time()-t)
            #pred_map = cv2.resize(pred_map, (sz[0], sz[1]))
            # print(9, time.time()-t)

            pred_map = torch.FloatTensor(blur(pred_map))
            # print(10, time.time()-t)
            grid = utils.make_grid(pred_map, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0)
            # print(11, time.time()-t)
            pred_map = torch.round(pred_map.mul(255).add_(0.5).clamp_(0, 255)).to('cpu', torch.uint8).numpy()
            # print(12, time.time()-t)
            #print(type(pred_map), type(pred_map[0]), type(pred_map[0][0]), type(pred_map[0][0][0]))
            # img_save(pred_map, "../results/sample.png", normalize=True)
            return pred_map
a = Image.open('../images/220921.jpg').convert('RGB')
sal = SaliencyInference()
for i in tqdm(range(300)):
    q = sal.get_saliency_map(a,0)