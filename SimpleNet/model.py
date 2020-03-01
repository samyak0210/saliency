import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../PNAS/')
from PNASnet import *
from genotypes import PNASNet

class PNASModel(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1):
        super(PNASModel, self).__init__()
        self.path = '../PNAS/PNASNet-5_Large.pth'

        self.pnas = NetworkImageNet(216, 1001, 12, False, PNASNet)
        if load_weight:
            self.pnas.load_state_dict(torch.load(self.path))
        
        for param in self.pnas.parameters():
            param.requires_grad = train_enc

        self.padding = nn.ConstantPad2d((0,1,0,1),0)
        self.drop_path_prob = 0

        self.linear_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.deconv_layer0 = nn.Sequential(
            nn.Conv2d(in_channels = 4320, out_channels = 512, kernel_size=3, padding=1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )

        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 512+2160, out_channels = 256, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 1080+256, out_channels = 270, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 540, out_channels = 96, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels = 192, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 128, out_channels = 1, kernel_size = 3, padding = 1, bias = True),
            nn.Sigmoid()
        )
    
    def forward(self, images):
        batch_size = images.size(0)

        s0 = self.pnas.conv0(images)
        s0 = self.pnas.conv0_bn(s0)
        out1 = self.padding(s0)

        s1 = self.pnas.stem1(s0, s0, self.drop_path_prob)
        out2 = s1
        s0, s1 = s1, self.pnas.stem2(s0, s1, 0)

        for i, cell in enumerate(self.pnas.cells):
            s0, s1 = s1, cell(s0, s1, 0)
            if i==3:
                out3 = s1
            if i==7:
                out4 = s1
            if i==11:
                out5 = s1
        
        out5 = self.deconv_layer0(out5)

        x = torch.cat((out5,out4), 1)
        x = self.deconv_layer1(x)
        
        x = torch.cat((x,out3), 1)
        x = self.deconv_layer2(x)

        x = torch.cat((x,out2), 1)
        x = self.deconv_layer3(x)
        
        x = torch.cat((x,out1), 1)
        x = self.deconv_layer4(x)
        x = self.deconv_layer5(x)
        x = x.squeeze(1)
        return x

class DenseModel(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1):
        super(DenseModel, self).__init__()

        self.dense = models.densenet161(pretrained=bool(load_weight)).features

        for param in self.dense.parameters():
            param.requires_grad = train_enc

        self.linear_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_layer0 = nn.Sequential(*list(self.dense)[:3])
        self.conv_layer1 = nn.Sequential(
        	self.dense.pool0,
        	self.dense.denseblock1,
        	*list(self.dense.transition1)[:3]
        )
        self.conv_layer2 = nn.Sequential(
        	self.dense.transition1[3],
        	self.dense.denseblock2,
        	*list(self.dense.transition2)[:3]
        )
        self.conv_layer3 = nn.Sequential(
        	self.dense.transition2[3],
        	self.dense.denseblock3,
        	*list(self.dense.transition3)[:3]
        )
        self.conv_layer4 = nn.Sequential(
        	self.dense.transition3[3],
        	self.dense.denseblock4
        )


        self.deconv_layer0 = nn.Sequential(
            nn.Conv2d(in_channels = 2208, out_channels = 512, kernel_size=3, padding=1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )

        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 512+1056, out_channels = 256, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 384+256, out_channels = 192, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 192+192, out_channels = 96, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels = 96+96, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 128, out_channels = 1, kernel_size = 3, padding = 1, bias = True),
            nn.Sigmoid()
        )

    def forward(self, images):
        batch_size = images.size(0)

        out1 = self.conv_layer0(images)
        out2 = self.conv_layer1(out1)
        out3 = self.conv_layer2(out2)
        out4 = self.conv_layer3(out3)
        out5 = self.conv_layer4(out4)

        
        assert out1.size() == (batch_size, 96, 128, 128)
        assert out2.size() == (batch_size, 192, 64, 64)
        assert out3.size() == (batch_size, 384, 32, 32)
        assert out4.size() == (batch_size, 1056, 16, 16)
        assert out5.size() == (batch_size, 2208, 8, 8)

        out5 = self.deconv_layer0(out5)

        x = torch.cat((out5,out4), 1)
        x = self.deconv_layer1(x)

        x = torch.cat((x,out3), 1)
        x = self.deconv_layer2(x)

        x = torch.cat((x,out2), 1)
        x = self.deconv_layer3(x)
        
        x = torch.cat((x,out1), 1)
        x = self.deconv_layer4(x)
        x = self.deconv_layer5(x)
        x = x.squeeze(1)
        return x

class ResNetModel(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1):
        super(ResNetModel, self).__init__()

        self.num_channels = num_channels
        self.resnet = models.resnet50(pretrained=bool(load_weight))
        
        for param in self.resnet.parameters():
            param.requires_grad = train_enc
        
        self.conv_layer1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu
        )
        self.conv_layer2 = nn.Sequential(
            self.resnet.maxpool,
            self.resnet.layer1
        )
        self.conv_layer3 = self.resnet.layer2
        self.conv_layer4 = self.resnet.layer3
        self.conv_layer5 = self.resnet.layer4

        self.linear_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv_layer0 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 2048, out_channels = 512, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 1024, out_channels = 256, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 64, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = 3, padding = 1, bias = True),
            nn.Sigmoid()
        )
    
    def forward(self, images):
        batch_size = images.size(0)

        out1 = self.conv_layer1(images)
        out2 = self.conv_layer2(out1)
        out3 = self.conv_layer3(out2)
        out4 = self.conv_layer4(out3)
        out5 = self.conv_layer5(out4)

        out5 = self.deconv_layer0(out5)
        assert out5.size() == (batch_size, 1024, 16, 16)

        x = torch.cat((out5,out4), 1)
        assert x.size() == (batch_size, 2048, 16, 16)
        x = self.deconv_layer1(x)
        assert x.size() == (batch_size, 512, 32, 32)
        
        x = torch.cat((x, out3), 1)
        assert x.size() == (batch_size, 1024, 32, 32)
        x = self.deconv_layer2(x)
        assert x.size() == (batch_size, 256, 64, 64)

        x = torch.cat((x, out2), 1)
        assert x.size() == (batch_size, 512, 64, 64)
        x = self.deconv_layer3(x)
        assert x.size() == (batch_size, 64, 128, 128)
        
        x = torch.cat((x, out1), 1)
        assert x.size() == (batch_size, 128, 128, 128)
        x = self.deconv_layer4(x)
        x = self.deconv_layer5(x)
        assert x.size() == (batch_size, 1, 256, 256)
        x = x.squeeze(1)
        assert x.size() == (batch_size, 256, 256)
        return x

class VGGModel(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1):
        super(VGGModel, self).__init__()

        self.num_channels = num_channels
        self.vgg = models.vgg16(pretrained=bool(load_weight)).features
        
        for param in self.vgg.parameters():
            param.requires_grad = train_enc
        
        self.conv_layer1 = self.vgg[:7]
        self.conv_layer2 = self.vgg[7:12]
        self.conv_layer3 = self.vgg[12:19]
        self.conv_layer4 = self.vgg[19:24]
        self.conv_layer5 = self.vgg[24:]

        self.linear_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 1024, out_channels = 512, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 1024, out_channels = 256, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 128, out_channels = 1, kernel_size = 3, padding = 1, bias = True),
            nn.Sigmoid()
        )
    
    def forward(self, images):
        batch_size = images.size(0)

        out1 = self.conv_layer1(images)
        out2 = self.conv_layer2(out1)
        out3 = self.conv_layer3(out2)
        out4 = self.conv_layer4(out3)
        out5 = self.conv_layer5(out4)

        out5 = self.linear_upsampling(out5)
        assert out5.size() == (batch_size, 512, 16, 16)

        x = torch.cat((out5,out4), 1)
        assert x.size() == (batch_size, 1024, 16, 16)
        x = self.deconv_layer1(x)
        assert x.size() == (batch_size, 512, 32, 32)
        
        x = torch.cat((x, out3), 1)
        assert x.size() == (batch_size, 1024, 32, 32)
        x = self.deconv_layer2(x)
        assert x.size() == (batch_size, 256, 64, 64)

        x = torch.cat((x, out2), 1)
        assert x.size() == (batch_size, 512, 64, 64)
        x = self.deconv_layer3(x)
        assert x.size() == (batch_size, 128, 128, 128)
        
        x = torch.cat((x, out1), 1)
        assert x.size() == (batch_size, 256, 128, 128)
        x = self.deconv_layer4(x)
        x = self.deconv_layer5(x)
        assert x.size() == (batch_size, 1, 256, 256)
        x = x.squeeze(1)
        assert x.size() == (batch_size, 256, 256)
        return x

