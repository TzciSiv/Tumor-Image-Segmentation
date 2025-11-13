import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class UNet(nn.Module):
    def __init__(self, num_classes=2, backbone='resnet50', pretrained=True):
        # super(ResNetUNet, self).__init__()
        super().__init__()

        """


        TO DO


        """
        # encoder
        # 1) 256x256 -> 128x128
        self.e11 = nn.Conv2d(in_channels=3,  out_channels=64, kernel_size=3, padding="same")
        self.bn11 = nn.BatchNorm2d(64)
        self.relu11 = nn.ReLU(inplace=True)
        
        self.e12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same")
        self.bn12 = nn.BatchNorm2d(64)
        self.relu12 = nn.ReLU(inplace=True)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2) 128x128 -> 64x64
        self.e21 = nn.Conv2d(64, 128, 3, padding="same")
        self.bn21 = nn.BatchNorm2d(128)
        self.relu21 = nn.ReLU(inplace=True)
        
        self.e22 = nn.Conv2d(128, 128, 3, padding="same")
        self.bn22 = nn.BatchNorm2d(128)
        self.relu22 = nn.ReLU(inplace=True)
        
        self.pool2 = nn.MaxPool2d(2, 2)

        # 3) 64x64 -> 32x32
        self.e31 = nn.Conv2d(128, 256, 3, padding="same")
        self.bn31 = nn.BatchNorm2d(256)
        self.relu31 = nn.ReLU(inplace=True)
        
        self.e32 = nn.Conv2d(256, 256, 3, padding="same")
        self.bn32 = nn.BatchNorm2d(256)
        self.relu32 = nn.ReLU(inplace=True)
        
        self.pool3 = nn.MaxPool2d(2, 2)

        # 4) bottleneck at 32x32
        self.e41 = nn.Conv2d(256, 512, 3, padding="same")
        self.bn41 = nn.BatchNorm2d(512)
        self.relu41 = nn.ReLU(inplace=True)
        
        self.e42 = nn.Conv2d(512, 512, 3, padding="same")
        self.relu42 = nn.ReLU(inplace=True)
        self.bn42 = nn.BatchNorm2d(512)


        # Decoder
        self.relu1 = nn.ReLU(inplace=True)
        
        # up from 32x32 -> 64x64, concat with encoder level-3 (256 ch)
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d11  = nn.Conv2d(256 + 256, 256, 3, padding="same")
        self.dbn11 = nn.BatchNorm2d(256)
        self.drelu11 = nn.ReLU(inplace=True)
        self.d12  = nn.Conv2d(256, 256, 3, padding="same")
        self.dbn12 = nn.BatchNorm2d(256)
        self.drelu12 = nn.ReLU(inplace=True)


        # up from 64x64 -> 128x128, concat with encoder level-2 (128 ch)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d21  = nn.Conv2d(128 + 128, 128, 3, padding="same")
        self.dbn21 = nn.BatchNorm2d(128)
        self.drelu21 = nn.ReLU(inplace=True)
        self.d22  = nn.Conv2d(128, 128, 3, padding="same")
        self.dbn22 = nn.BatchNorm2d(128)
        self.drelu22 = nn.ReLU(inplace=True)

        # up from 128x128 -> 256x256, concat with encoder level-1 (64 ch)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d31  = nn.Conv2d(64 + 64, 64, 3, padding="same")
        self.dbn31 = nn.BatchNorm2d(64)
        self.drelu31 = nn.ReLU(inplace=True)
        self.d32  = nn.Conv2d(64, 64, 3, padding="same")
        self.dbn32 = nn.BatchNorm2d(64)
        self.drelu32 = nn.ReLU(inplace=True)

        # Output layer
        self.outconv = nn.Conv2d(64, num_classes, kernel_size=1)


    def forward(self, x):
        # Encoder
        """


        TO DO


        """
        x1 = self.relu11(self.bn11(self.e11(x)))
        x1 = self.relu12(self.bn12(self.e12(x1)))
        p1 = self.pool1(x1)

        x2 = self.relu21(self.bn21(self.e21(p1)))
        x2 = self.relu22(self.bn22(self.e22(x2)))
        p2 = self.pool2(x2)

        x3 = self.relu31(self.bn31(self.e31(p2)))
        x3 = self.relu32(self.bn32(self.e32(x3)))
        p3 = self.pool3(x3)

        b  = self.relu41(self.bn41(self.e41(p3)))
        b  = self.relu42(self.bn42(self.e42(b)))   # bottleneck (32x32, 512 ch)


        # Decoder
        """


        TO DO


        """
        # 1) 32->64, concat with xe32 (256 ch)
        u1 = self.upconv1(b)
        y1 = torch.cat([u1, x3], dim=1)
        y1 = self.relu1(self.dbn11(self.d11(y1)))
        y1 = self.relu1(self.dbn12(self.d12(y1)))

        # 2) 64->128, concat with xe22 (128 ch)
        u2 = self.upconv2(y1)
        y2 = torch.cat([u2, x2], dim=1)
        y2 = self.relu1(self.dbn21(self.d21(y2)))
        y2 = self.relu1(self.dbn22(self.d22(y2)))

        # 3) 128->256, concat with xe12 (64 ch)
        u3 = self.upconv3(y2)
        y3 = torch.cat([u3, x1], dim=1)
        y3 = self.relu1(self.dbn31(self.d31(y3)))
        y3 = self.relu1(self.dbn32(self.d32(y3)))

        # 🟩 Final upsampling step
        """


        TO DO


        """
        out = self.outconv(y3)
        
        return out
    
class ResUNet1(nn.Module):
    def __init__(self, num_classes=2, backbone='resnet50', pretrained=True):
        super().__init__()

        """
        ResUNet 1:
            - ResNet50 encoder (FROZEN)
            - UNet-style decoder
            - Output: raw logits [B, num_classes, H, W]
        """

        # encoder (Frozen ResNet50)
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet = models.resnet50(weights=weights) 

        # Freeze encoder parameters
        for p in self.resnet.parameters():
            p.requires_grad = False

        # Keep BN in eval mode
        self.resnet.eval()

        # Decoder (UNet-style) 
        self.relu1 = nn.ReLU(inplace=True)

        # up from 1/32 -> 1/16, concat with enc_l3 (1024 ch)
        self.upconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.d11  = nn.Conv2d(1024 + 1024, 1024, 3, padding=1)
        self.dbn11 = nn.BatchNorm2d(1024)
        self.drelu11 = nn.ReLU(inplace=True)
        self.d12  = nn.Conv2d(1024, 1024, 3, padding=1)
        self.dbn12 = nn.BatchNorm2d(1024)
        self.drelu12 = nn.ReLU(inplace=True)

        # up from 1/16 -> 1/8, concat with enc_l2 (512 ch)
        self.upconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d21  = nn.Conv2d(512 + 512, 512, 3, padding=1)
        self.dbn21 = nn.BatchNorm2d(512)
        self.drelu21 = nn.ReLU(inplace=True)
        self.d22  = nn.Conv2d(512, 512, 3, padding=1)
        self.dbn22 = nn.BatchNorm2d(512)
        self.drelu22 = nn.ReLU(inplace=True)

        # up from 1/8 -> 1/4, concat with enc_l1 (256 ch)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d31  = nn.Conv2d(256 + 256, 256, 3, padding=1)
        self.dbn31 = nn.BatchNorm2d(256)
        self.drelu31 = nn.ReLU(inplace=True)
        self.d32  = nn.Conv2d(256, 256, 3, padding=1)
        self.dbn32 = nn.BatchNorm2d(256)
        self.drelu32 = nn.ReLU(inplace=True)

        # up from 1/4 -> 1/2, concat with stem (64 ch)
        self.upconv4 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.d41  = nn.Conv2d(64 + 64, 64, 3, padding=1)
        self.dbn41 = nn.BatchNorm2d(64)
        self.drelu41 = nn.ReLU(inplace=True)
        self.d42  = nn.Conv2d(64, 64, 3, padding=1)
        self.dbn42 = nn.BatchNorm2d(64)
        self.drelu42 = nn.ReLU(inplace=True)

        # final up from 1/2 -> 1/1
        self.upconv5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.d51 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )

        # Output head
        self.outconv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.resnet.relu(self.resnet.bn1(self.resnet.conv1(x)))    # 64, 1/2
        x1_in = self.resnet.maxpool(x0)                                 # 1/4
        x1 = self.resnet.layer1(x1_in)                                  # 256, 1/4
        x2 = self.resnet.layer2(x1)                                     # 512, 1/8
        x3 = self.resnet.layer3(x2)                                     # 1024, 1/16
        x4 = self.resnet.layer4(x3)                                     # 2048, 1/32

        # Decoder
        # 1) 1/32 -> 1/16, concat with x3
        u1 = self.upconv1(x4)
        y1 = torch.cat([u1, x3], dim=1)
        y1 = self.drelu11(self.dbn11(self.d11(y1)))
        y1 = self.drelu12(self.dbn12(self.d12(y1)))

        # 2) 1/16 -> 1/8, concat with x2
        u2 = self.upconv2(y1)
        y2 = torch.cat([u2, x2], dim=1)
        y2 = self.drelu21(self.dbn21(self.d21(y2)))
        y2 = self.drelu22(self.dbn22(self.d22(y2)))

        # 3) 1/8 -> 1/4, concat with x1
        u3 = self.upconv3(y2)
        y3 = torch.cat([u3, x1], dim=1)
        y3 = self.drelu31(self.dbn31(self.d31(y3)))
        y3 = self.drelu32(self.dbn32(self.d32(y3)))

        # 4) 1/4 -> 1/2, concat with x0
        u4 = self.upconv4(y3)
        y4 = torch.cat([u4, x0], dim=1)
        y4 = self.drelu41(self.dbn41(self.d41(y4)))
        y4 = self.drelu42(self.dbn42(self.d42(y4)))

        # 5) 1/2 -> 1/1
        u5 = self.upconv5(y4)
        y5 = self.d51(u5)

        # Output layer
        out = self.outconv(y5)

        return out
    
class ResUNet2(nn.Module):
    def __init__(self, num_classes=2, backbone='resnet50', pretrained=True):
        super().__init__()

        """
        ResUNet 2:
            - ResNet50 encoder (FINE-TUNE ALL: encoder + decoder are trainable)
            - UNet-style decoder
            - Output: raw logits [B, num_classes, H, W]
        """
        # ----- Encoder (ResNet50, trainable) -----
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet = models.resnet50(weights=weights)   # all params default to requires_grad=True

        # ----- Decoder (UNet-style) -----
        # up from 1/32 -> 1/16, concat with enc_l3 (1024 ch)
        self.upconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.d11  = nn.Conv2d(1024 + 1024, 1024, 3, padding=1)
        self.dbn11 = nn.BatchNorm2d(1024)
        self.drelu11 = nn.ReLU(inplace=True)
        self.d12  = nn.Conv2d(1024, 1024, 3, padding=1)
        self.dbn12 = nn.BatchNorm2d(1024)
        self.drelu12 = nn.ReLU(inplace=True)

        # up from 1/16 -> 1/8, concat with enc_l2 (512 ch)
        self.upconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d21  = nn.Conv2d(512 + 512, 512, 3, padding=1)
        self.dbn21 = nn.BatchNorm2d(512)
        self.drelu21 = nn.ReLU(inplace=True)
        self.d22  = nn.Conv2d(512, 512, 3, padding=1)
        self.dbn22 = nn.BatchNorm2d(512)
        self.drelu22 = nn.ReLU(inplace=True)

        # up from 1/8 -> 1/4, concat with enc_l1 (256 ch)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d31  = nn.Conv2d(256 + 256, 256, 3, padding=1)
        self.dbn31 = nn.BatchNorm2d(256)
        self.drelu31 = nn.ReLU(inplace=True)
        self.d32  = nn.Conv2d(256, 256, 3, padding=1)
        self.dbn32 = nn.BatchNorm2d(256)
        self.drelu32 = nn.ReLU(inplace=True)

        # up from 1/4 -> 1/2, concat with stem (64 ch)
        self.upconv4 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.d41  = nn.Conv2d(64 + 64, 64, 3, padding=1)
        self.dbn41 = nn.BatchNorm2d(64)
        self.drelu41 = nn.ReLU(inplace=True)
        self.d42  = nn.Conv2d(64, 64, 3, padding=1)
        self.dbn42 = nn.BatchNorm2d(64)
        self.drelu42 = nn.ReLU(inplace=True)

        # final up from 1/2 -> 1/1
        self.upconv5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.d51 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Output head
        self.outconv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # ----- Encoder -----
        x0 = self.resnet.relu(self.resnet.bn1(self.resnet.conv1(x)))  # 64, 1/2
        x1_in = self.resnet.maxpool(x0)                               # 1/4
        x1 = self.resnet.layer1(x1_in)                                # 256, 1/4
        x2 = self.resnet.layer2(x1)                                   # 512, 1/8
        x3 = self.resnet.layer3(x2)                                   # 1024, 1/16
        x4 = self.resnet.layer4(x3)                                   # 2048, 1/32

        # ----- Decoder -----
        # 1) 1/32 -> 1/16, concat with x3
        u1 = self.upconv1(x4)
        y1 = torch.cat([u1, x3], dim=1)
        y1 = self.drelu11(self.dbn11(self.d11(y1)))
        y1 = self.drelu12(self.dbn12(self.d12(y1)))

        # 2) 1/16 -> 1/8, concat with x2
        u2 = self.upconv2(y1)
        y2 = torch.cat([u2, x2], dim=1)
        y2 = self.drelu21(self.dbn21(self.d21(y2)))
        y2 = self.drelu22(self.dbn22(self.d22(y2)))

        # 3) 1/8 -> 1/4, concat with x1
        u3 = self.upconv3(y2)
        y3 = torch.cat([u3, x1], dim=1)
        y3 = self.drelu31(self.dbn31(self.d31(y3)))
        y3 = self.drelu32(self.dbn32(self.d32(y3)))

        # 4) 1/4 -> 1/2, concat with x0
        u4 = self.upconv4(y3)
        y4 = torch.cat([u4, x0], dim=1)
        y4 = self.drelu41(self.dbn41(self.d41(y4)))
        y4 = self.drelu42(self.dbn42(self.d42(y4)))

        # 5) 1/2 -> 1/1
        u5 = self.upconv5(y4)
        y5 = self.d51(u5)

        # Output logits
        return self.outconv(y5)
    
class ResUNet3(nn.Module): 
    def __init__(self, num_classes=2, backbone='resnet50', pretrained=True):
        super().__init__()

        """
        ResUNet 3:
            - ResNet50 encoder (PARTIAL FINE-TUNE)
                • conv1, bn1, layer1–3 are frozen
                • layer4 remains trainable
            - Output: raw logits [B, num_classes, H, W]
        """
        # ----- Encoder -----
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet = models.resnet50(weights=weights)

        # Freeze early stages: conv1, bn1, layer1-3
        for m in [self.resnet.conv1, self.resnet.bn1, self.resnet.layer1, self.resnet.layer2, self.resnet.layer3]:
            for p in m.parameters():
                p.requires_grad = False
            # keep BN in eval for frozen parts
            for mm in m.modules():
                if isinstance(mm, nn.BatchNorm2d):
                    mm.eval()

        # layer4 stays trainable (no freezing)

        # ----- Decoder (trainable) -----
        self.upconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.d11  = nn.Conv2d(1024 + 1024, 1024, 3, padding=1)
        self.dbn11 = nn.BatchNorm2d(1024)
        self.drelu11 = nn.ReLU(inplace=True)
        self.d12  = nn.Conv2d(1024, 1024, 3, padding=1)
        self.dbn12 = nn.BatchNorm2d(1024)
        self.drelu12 = nn.ReLU(inplace=True)

        self.upconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d21  = nn.Conv2d(512 + 512, 512, 3, padding=1)
        self.dbn21 = nn.BatchNorm2d(512)
        self.drelu21 = nn.ReLU(inplace=True)
        self.d22  = nn.Conv2d(512, 512, 3, padding=1)
        self.dbn22 = nn.BatchNorm2d(512)
        self.drelu22 = nn.ReLU(inplace=True)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d31  = nn.Conv2d(256 + 256, 256, 3, padding=1)
        self.dbn31 = nn.BatchNorm2d(256)
        self.drelu31 = nn.ReLU(inplace=True)
        self.d32  = nn.Conv2d(256, 256, 3, padding=1)
        self.dbn32 = nn.BatchNorm2d(256)
        self.drelu32 = nn.ReLU(inplace=True)

        self.upconv4 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.d41  = nn.Conv2d(64 + 64, 64, 3, padding=1)
        self.dbn41 = nn.BatchNorm2d(64)
        self.drelu41 = nn.ReLU(inplace=True)
        self.d42  = nn.Conv2d(64, 64, 3, padding=1)
        self.dbn42 = nn.BatchNorm2d(64)
        self.drelu42 = nn.ReLU(inplace=True)

        self.upconv5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.d51 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.outconv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0    = self.resnet.relu(self.resnet.bn1(self.resnet.conv1(x)))  # 64, 1/2
        x1_in = self.resnet.maxpool(x0)                                  # 1/4
        x1    = self.resnet.layer1(x1_in)                                # 256, 1/4 (frozen)
        x2    = self.resnet.layer2(x1)                                   # 512, 1/8 (frozen)
        x3    = self.resnet.layer3(x2)                                   # 1024, 1/16 (frozen)
        x4    = self.resnet.layer4(x3)                                   # 2048, 1/32 (trainable)

        # Decoder
        u1 = self.upconv1(x4)                 # -> 1/16
        y1 = torch.cat([u1, x3], dim=1)
        y1 = self.drelu11(self.dbn11(self.d11(y1)))
        y1 = self.drelu12(self.dbn12(self.d12(y1)))

        u2 = self.upconv2(y1)                 # -> 1/8
        y2 = torch.cat([u2, x2], dim=1)
        y2 = self.drelu21(self.dbn21(self.d21(y2)))
        y2 = self.drelu22(self.dbn22(self.d22(y2)))

        u3 = self.upconv3(y2)                 # -> 1/4
        y3 = torch.cat([u3, x1], dim=1)
        y3 = self.drelu31(self.dbn31(self.d31(y3)))
        y3 = self.drelu32(self.dbn32(self.d32(y3)))

        u4 = self.upconv4(y3)                 # -> 1/2
        y4 = torch.cat([u4, x0], dim=1)
        y4 = self.drelu41(self.dbn41(self.d41(y4)))
        y4 = self.drelu42(self.dbn42(self.d42(y4)))

        u5 = self.upconv5(y4)                 # -> 1/1
        y5 = self.d51(u5)

        return self.outconv(y5)