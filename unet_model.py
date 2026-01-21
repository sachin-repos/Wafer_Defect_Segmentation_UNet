# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 20:47:32 2026

@author: USER
"""

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Conv2d => BatchNorm => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # --- ENCODER (Downsampling) ---
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        # --- BOTTLENECK ---
        self.bottleneck = DoubleConv(256, 512)
        
        # --- DECODER (Upsampling) ---
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up_conv3 = DoubleConv(512, 256) # 512 because we concatenate
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_conv2 = DoubleConv(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_conv1 = DoubleConv(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Down
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        
        # Bottom
        b = self.bottleneck(p3)
        
        # Up (With Skip Connections - The "U" shape)
        u3 = self.up3(b)
        u3 = torch.cat((u3, d3), dim=1) # Concatenate Skip Connection
        x = self.up_conv3(u3)
        
        u2 = self.up2(x)
        u2 = torch.cat((u2, d2), dim=1)
        x = self.up_conv2(u2)
        
        u1 = self.up1(x)
        u1 = torch.cat((u1, d1), dim=1)
        x = self.up_conv1(u1)
        
        return self.final_conv(x)