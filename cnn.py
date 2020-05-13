#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, char_embed_size, word_embed_size, kernel_size=5):
        super(CNN, self).__init__()
        self.conv_layer = nn.Conv1d(in_channels=char_embed_size,
                                    out_channels=word_embed_size,
                                    kernel_size=kernel_size)

    def forward(self, x_reshaped):
        x_conv = self.conv_layer(x_reshaped)
        x_conv_out = torch.max(F.relu(x_conv), dim=2)[0]
        return x_conv_out
