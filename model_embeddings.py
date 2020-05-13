#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(f)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
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

        self.char_embed_size = 50
        self.embed_size = embed_size
        self.char_embedding = nn.Embedding(len(vocab.char2id), self.char_embed_size, padding_idx=0)
        self.cnn = CNN(char_embed_size=self.char_embed_size, word_embed_size=embed_size, kernel_size=5)
        self.highway = Highway(word_embed_size=embed_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input_tensor: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        x_char_embed = self.char_embedding(input_tensor)
        # shape: (sentence_length, batch_size, max_word_length, e_char)

        x_reshaped = x_char_embed.permute(0, 1, 3, 2)
        # shape: (sentence_length, batch_size, e_char, max_word_length)

        x_conv = self.cnn(x_reshaped.view(-1, self.char_embed_size, input_tensor.shape[2]))
        # (seq_len*batch_size, e_word)

        x_highway = self.highway(x_conv)
        # shape: (batch_size*seq_len, e_word)

        output = self.dropout(x_highway.view(input_tensor.shape[0], input_tensor.shape[1], self.embed_size))
        # shape: sentence_length, batch_size, embed_size

        return output
