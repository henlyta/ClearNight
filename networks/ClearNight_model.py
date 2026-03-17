import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_

import torch.nn.init as init
import matplotlib.pyplot as plt
import os
import torchvision.utils as vutils
import cv2
from .baseblocks import *
from .DSM import MixtureOfUnits

all_labels = ["haze", "snow", "rain", "drop"]
num_labels = len(all_labels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DWS(nn.Module):
    def __init__(self, dim, num_labels): 
        super(DWS, self).__init__()

        self.DWS_encoder = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=2, dilation=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(dim // 2, 12, kernel_size=3, padding=2, dilation=2, stride=2),
        )

        self.moe_model = MixtureOfUnits(input_channels=12, output_channels=12,
                                          num_units=25, top_k=10, num_labels=num_labels)

        self.DWS_decoder = nn.Sequential(
            nn.ConvTranspose2d(12, dim // 2, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(dim // 2, dim // 2, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(dim // 2, dim, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.DWS_decoder:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                init.constant_(layer.weight, 0)
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)

        for layer in self.DWS_encoder:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                init.normal_(layer.weight, mean=0.0, std=0.02)
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)

    def forward(self, x, rmap, file_name=None):
        z = torch.cat([x, rmap], dim=1)
        z = self.DWS_encoder(z)
        z, classification_logits, l2_regularization, expert_indices, load_balancing_loss = self.moe_model(z, file_name=file_name)
        z = self.DWS_decoder(z)
        return z, classification_logits, l2_regularization, expert_indices, load_balancing_loss

class ClearNight(nn.Module):
    def __init__(self, in_chans=3, out_chans=4, window_size=8,
                 embed_dims=[24, 48, 96, 48, 24],
                 mlp_ratios=[2., 4., 4., 2., 2.],
                 depths=[16, 16, 16, 8, 8],
                 num_heads=[2, 4, 6, 1, 1],
                 attn_ratio=[1/4, 1/2, 3/4, 0, 0],
                 conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
                 norm_layer=[RLN, RLN, RLN, RLN, RLN],
                 num_labels=4): 
        super(ClearNight, self).__init__()

        self.patch_size = 4
        self.window_size = window_size
        self.mlp_ratios = mlp_ratios

        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)

        self.layer1 = BasicLayer(network_depth=sum(depths), dim=embed_dims[0], depth=depths[0],
                                 num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                                 norm_layer=norm_layer[0], window_size=window_size,
                                 attn_ratio=attn_ratio[0], attn_loc='last', conv_type=conv_type[0])

        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])

        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.layer2 = BasicLayer(network_depth=sum(depths), dim=embed_dims[1], depth=depths[1],
                                 num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                                 norm_layer=norm_layer[1], window_size=window_size,
                                 attn_ratio=attn_ratio[1], attn_loc='last', conv_type=conv_type[1])

        self.patch_merge2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.layer3 = BasicLayer(network_depth=sum(depths), dim=embed_dims[2], depth=depths[2],
                                 num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                                 norm_layer=norm_layer[2], window_size=window_size,
                                 attn_ratio=attn_ratio[2], attn_loc='last', conv_type=conv_type[2])

        self.patch_split1 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = SKFusion(embed_dims[3])

        self.layer4 = BasicLayer(network_depth=sum(depths), dim=embed_dims[3], depth=depths[3],
                                 num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                                 norm_layer=norm_layer[3], window_size=window_size,
                                 attn_ratio=attn_ratio[3], attn_loc='last', conv_type=conv_type[3])

        self.patch_split2 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = SKFusion(embed_dims[4])

        self.layer5 = BasicLayer(network_depth=sum(depths), dim=embed_dims[4], depth=depths[4],
                                 num_heads=num_heads[4], mlp_ratio=mlp_ratios[4],
                                 norm_layer=norm_layer[4], window_size=window_size,
                                 attn_ratio=attn_ratio[4], attn_loc='last', conv_type=conv_type[4])

        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)

        self.MPE = MPE(channel=48)

        self.moe1 = DWS(embed_dims[0], num_labels)
        self.moe2 = DWS(embed_dims[1], num_labels)
        self.moe3 = DWS(embed_dims[2], num_labels)
        self.moe4 = DWS(embed_dims[3], num_labels)
        self.moe5 = DWS(embed_dims[4], num_labels)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x    

    def forward_features(self, x, imap, rmap, file_name=None):

        imap1, imap2, imap3 = self.MPE(imap)
        rmap1, rmap2, rmap3 = self.MPE(rmap)

        x = self.patch_embed(x)
        y1, cls_logits1, l2_reg1, expert_indices1, load_balancing_loss1 = self.moe1(x, rmap1, file_name=file_name)
        x = self.layer1(x, imap=imap1)
        x = x + y1
        skip1 = x

        x = self.patch_merge1(x)
        y2, cls_logits2, l2_reg2, expert_indices2, load_balancing_loss2 = self.moe2(x, rmap2, file_name=file_name)
        x = self.layer2(x, imap=imap2)
        x = x + y2
        skip2 = x

        x = self.patch_merge2(x)
        y3, cls_logits3, l2_reg3, expert_indices3, load_balancing_loss3 = self.moe3(x, rmap3, file_name=file_name)
        x = self.layer3(x, imap=imap3)
        x = x + y3
        x = self.patch_split1(x)

        x = self.fusion1([x, self.skip2(skip2)]) + x
        y4, cls_logits4, l2_reg4, expert_indices4, load_balancing_loss4 = self.moe4(x, rmap2, file_name=file_name)
        x = self.layer4(x, imap=None)
        x = x + y4
        x = self.patch_split2(x)

        x = self.fusion2([x, self.skip1(skip1)]) + x
        y5, cls_logits5, l2_reg5, expert_indices5, load_balancing_loss5 = self.moe5(x, rmap1, file_name=file_name)
        x = self.layer5(x, imap=None)
        x = x + y5
        x = self.patch_unembed(x)

        classification_logits = cls_logits1 + cls_logits2 + cls_logits3 + cls_logits4 + cls_logits5
        l2_regularization = l2_reg1 + l2_reg2 + l2_reg3 + l2_reg4 + l2_reg5
        load_balancing_loss = load_balancing_loss1 + load_balancing_loss2 + load_balancing_loss3 + load_balancing_loss4 + load_balancing_loss5

        return x, classification_logits, l2_regularization, load_balancing_loss

    def forward(self, x, imap, rmap, file_name=None):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        feat, classification_logits, l2_regularization, load_balancing_loss = self.forward_features(x, imap, rmap, file_name)
        K, B = torch.split(feat, (1, 3), dim=1)
        x = K * x - B + x
        x = x[:, :, :H, :W]

        return x, classification_logits, l2_regularization, load_balancing_loss

def ClearNight_():
    return ClearNight(
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[8, 8, 8, 4, 4],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[1/4, 1/2, 3/4, 0, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
        num_labels=len(all_labels))

if __name__ == '__main__':
    model = ClearNight_()
    tmp = torch.randn(1, 3, 256, 256)
    file_name = None
    output, _, _, _ = model(tmp, tmp, tmp, file_name)
    print(output.shape)
