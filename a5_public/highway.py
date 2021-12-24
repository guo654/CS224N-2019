#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import torch.functional as F

class Highway(nn.Module):
    """
    map x_conv_out to an embedding vector
    """
    def __init__(self, word_embed_size):
        super(Highway, self).__init__()
        self.word_embed_size = word_embed_size
        self.proj = nn.Linear(self.word_embed_size, self.word_embed_size)
        self.gate = nn.Linear(self.word_embed_size, self.word_embed_size)

    def forward(self, x_conv_out: torch.Tensor) -> torch.Tensor:
        x_proj = nn.functional.relu(self.proj(x_conv_out))
        x_gate = torch.sigmoid(self.gate(x_conv_out))

        x = x_gate * x_proj + (1 - x_gate) * x_conv_out
        return x
### END YOUR CODE 

