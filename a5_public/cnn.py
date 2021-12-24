#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
"""
map x_reshaped to x_conv_out
"""
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np

class CNN(nn.Module):
    def __init__(self, num_filter, char_embed_size, max_word_length, kernel_size = 5):
        super(CNN, self).__init__()
        self.num_filter = num_filter
        self.char_embed_size = char_embed_size
        self.max_word_length = max_word_length
        self.kernel_size = kernel_size

        self.conv1d = nn.Conv1d(
            in_channels = self.char_embed_size,
            out_channels = self.num_filter,
            kernel_size = self.kernel_size,
            bias = True
        )

        self.max_pool_1d = nn.MaxPool1d(kernel_size = (self.max_word_length - self.kernel_size + 1))

    def forward(self, inpt):
        """
        input(Tensor) shape(batch_size, char_enbed_size, max_word_length)
        return Tensor: shape(batch_size, word_embed_size)
        """
        x = self.conv1d(inpt) # (batch_size, word_embed_size, max_word_length - kernel_size + 1)
        x = self.max_pool_1d(nn.functional.relu(x)).squeeze() # (batch_size, word_embed_size)
        return x

### END YOUR CODE

