#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):

    def __init__(self, word_embed_size):
        super().__init__()

        self.proj_layer = nn.Linear(in_features=word_embed_size,
                                    out_features=word_embed_size,
                                    bias=True)
        self.gate_layer = nn.Linear(in_features=word_embed_size,
                                    out_features=word_embed_size,
                                    bias=True)

        def forward(self, input: torch.Tensor):
            x_proj = F.relu(self.proj_layer(input))
            x_gate = nn.Sigmoid(self.gate_layer(input))
            x_highway = x_gate * x_proj + (1 - x_gate) * input
