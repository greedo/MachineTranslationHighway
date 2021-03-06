#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super().__init__()
        self.charDecoder = nn.LSTM(input_size=char_embedding_size,
                                   hidden_size=hidden_size)
        self.char_output_projection = nn.Linear(hidden_size,
                                                len(target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(num_embeddings=len(target_vocab.char2id),
                                           embedding_dim=char_embedding_size,
                                           padding_idx=target_vocab.char2id['<pad>'])
        self.target_vocab = target_vocab

    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        input_embed = self.decoderCharEmb(input)
        output, (h_n, c_n) = self.charDecoder(input_embed, dec_hidden)
        s_t = self.char_output_projection(output)
        return s_t, (h_n, c_n)

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch, for every character in the sequence.
        """
        s_t, (h_n, c_n) = self.forward(char_sequence[:-1], dec_hidden=dec_hidden)
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.target_vocab.char2id['<pad>'], reduction='sum') # ignore padding chars
        loss = loss_fn(s_t.view(-1, len(self.target_vocab.char2id)), char_sequence[1:].contiguous().view(-1))
        return loss

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        batch_size = initialStates[0].shape[1]

        current_char = [self.target_vocab.char2id['{']] * batch_size
        # print(batch_size == len(current_char))

        decodedWords = ['{'] * batch_size
        # print(batch_size == len(decodedWords))

        current_char_tensor = torch.tensor(current_char, device=device)
        # print(batch_size == current_char_tensor)

        dec_hidden = initialStates
        # initialize h_prev and c_prev to the given states from the LSTM

        for t in range(max_length):
            _, (dec_hidden) = self.forward(current_char_tensor.unsqueeze(0), (dec_hidden))
            s = self.char_output_projection(dec_hidden[0].squeeze(0))
            p = F.log_softmax(s, dim=1)
            current_char_tensor = torch.argmax(p, dim=1)

            if current_char == '}':
                break

            for i in range(batch_size):
                decodedWords[i] += self.target_vocab.id2char[current_char_tensor[i].item()]

        for i in range(batch_size):
            decodedWords[i] = decodedWords[i][1:]
            decodedWords[i] = decodedWords[i].partition('}')[0]

        return decodedWords
