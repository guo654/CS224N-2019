#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    x_padded to x_word_emb
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        self.embed_size = embed_size
        self.vocab = vocab
        self.max_word_length = 21
        self.char_embed_size = 50
        self.dropout_rate = 0.3

        self.char_embedding = nn.Embedding(
            num_embeddings=len(self.vocab.char2id),
            embedding_dim=self.char_embed_size,
            padding_idx=self.vocab.char2id['<pad>'],
        )

        self.CNN = CNN(
            num_filter=self.embed_size,
            char_embed_size=self.char_embed_size,
            max_word_length=self.max_word_length
        )

        self.Highway = Highway(
            word_embed_size=embed_size
        )
        self.dropout = nn.Dropout(self.dropout_rate)


        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        char_embeddings = self.char_embedding(input) # sentence_length, batch_size, max_word_length
        sentence_length, batch_size, _, _ = char_embeddings.shape
        char_embeddings = char_embeddings.view(sentence_length * batch_size, self.max_word_length, self.char_embed_size).transpose(1, 2)

        x_conv_out = self.CNN(char_embeddings)
        x_highway = self.Highway(x_conv_out)

        x_word_emb = self.dropout(x_highway)
        x_word_emb = x_word_emb.view(sentence_length, batch_size, self.embed_size)

        return x_word_emb

        ### END YOUR CODE

