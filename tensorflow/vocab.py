# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements the Vocab class for converting string to id and back
"""

import numpy as np


class Vocab(object):
    """
    Implements a vocabulary to store the tokens in the data, with their corresponding embeddings.
    """
    def __init__(self, filename=None, initial_tokens=None, lower=False):
        self.id2token = {}
        self.token2id = {}
        self.token_cnt = {}
        self.id2char = {}
        self.char2id = {}
        self.char_cnt = {}
        self.lower = lower

        self.embed_dim = None
        self.embeddings = None
        self.char_embed_dim = None
        self.char_embeddings = None

        self.pad_token = '<blank>'
        self.unk_token = '<unk>'
        self.pad_char = '<char_blank>'
        self.unk_char = '<char_unk>'

        self.initial_tokens = initial_tokens if initial_tokens is not None else []
        self.initial_tokens.extend([self.pad_token, self.unk_token])
        self.initial_chars = []
        self.initial_chars.extend([self.pad_char , self.unk_char])

        for token in self.initial_tokens:
            self.add(token)

        for char in self.initial_chars:
            self.add_char(char)

        if filename is not None:
            self.load_from_file(filename)

    def size(self):
        """
        get the size of vocabulary
        Returns:
            an integer indicating the size
        """
        return len(self.id2token)

    def char_size(self):
        return len(self.id2char)

    def load_from_file(self, file_path, use_char_level = False):
        """
        loads the vocab from file_path
        Args:
            file_path: a file with a word in each line
        """
        for line in open(file_path, 'r'):
            token = line.rstrip('\n')
            self.add(token)
            if use_char_level:
                for char in token:
                    self.add_char(char)

    def get_id(self, token):
        """
        gets the id of a token, returns the id of unk token if token is not in vocab
        Args:
            key: a string indicating the word
        Returns:
            an integer
        """
        token = token.lower() if self.lower else token
        try:
            return self.token2id[token]
        except KeyError:
            return self.token2id[self.unk_token]
    def get_char_id(self , char):
        try:
            return self.char2id[char]
        except KeyError:
            return self.char2id[self.unk_char]

    def get_token(self, idx):
        """
        gets the token corresponding to idx, returns unk token if idx is not in vocab
        Args:
            idx: an integer
        returns:
            a token string
        """
        try:
            return self.id2token[idx]
        except KeyError:
            return self.unk_token

    def get_char(self):
        try:
            return self.id2char[idx]
        except KeyError:
            return self.unk_char

    def add(self, token, cnt=1):
        """
        adds the token to vocab
        Args:
            token: a string
            cnt: a num indicating the count of the token to add, default is 1
        """
        token = token.lower() if self.lower else token
        if token in self.token2id:
            idx = self.token2id[token]
        else:
            idx = len(self.id2token)
            self.id2token[idx] = token
            self.token2id[token] = idx
        if cnt > 0:
            if token in self.token_cnt:
                self.token_cnt[token] += cnt
            else:
                self.token_cnt[token] = cnt
        return idx


    def add_char(self, char):
        if char in self.char2id:
            idx = self.char2id[char]
        else:
            idx = len(self.id2char)
            self.id2char[idx] = char
            self.char2id[char] = idx
        if char in self.char_cnt:
            self.char_cnt[char] += 1
        else :
            self.char_cnt[char] = 1
        return idx

    def filter_tokens_by_cnt(self, min_cnt):
        """
        filter the tokens in vocab by their count
        Args:
            min_cnt: tokens with frequency less than min_cnt is filtered
        """
        filtered_tokens = [token for token in self.token2id if self.token_cnt[token] >= min_cnt]
        # rebuild the token x id map
        self.token2id = {}
        self.id2token = {}
        for token in self.initial_tokens:
            self.add(token, cnt=0)
        for token in filtered_tokens:
            self.add(token, cnt=0)

    def filter_chars_by_cnt(self , min_cnt):
        filtered_chars = [char for char in self.char2id if self.char_cnt[char] >= min_cnt]
        self.char2id = {}
        self.id2char = {}
        for char in self.initial_chars:
            self.add_char(char)
        for char in filtered_chars:
            self.add_char(char)

    def randomly_init_embeddings(self, embed_dim):
        """
        randomly initializes the embeddings for each token
        Args:
            embed_dim: the size of the embedding for each token
        """
        self.embed_dim = embed_dim
        self.embeddings = np.random.rand(self.size(), embed_dim)
        for token in [self.pad_token, self.unk_token]:
            self.embeddings[self.get_id(token)] = np.zeros([self.embed_dim])

    def randomly_init_char_embeddings(self,char_embed_dim):
        self.char_embed_dim = char_embed_dim
        self.char_embeddings = np.random.rand(self.char_size() , char_embed_dim)
        for char in [self.pad_char , self.unk_char]:
            self.char_embeddings[self.get_char_id(char)] = np.zeros([self.char_embed_dim])


    def load_pretrained_embeddings(self, embedding_path):
        """
        loads the pretrained embeddings from embedding_path,
        tokens not in pretrained embeddings will be filtered
        Args:
            embedding_path: the path of the pretrained embedding file
        """
        trained_embeddings = {}
        with open(embedding_path, 'r') as fin:
            for line in fin:
                contents = line.strip().split()
                #token = contents[0].decode('utf8')#py2
                token = contents[0]#py3
                if token not in self.token2id:
                    continue
                trained_embeddings[token] = list(map(float, contents[1:]))
                if self.embed_dim is None:
                    self.embed_dim = len(contents) - 1
        filtered_tokens = trained_embeddings.keys()
        # rebuild the token x id map
        self.token2id = {}
        self.id2token = {}
        for token in self.initial_tokens:
            self.add(token, cnt=0)
        for token in filtered_tokens:
            self.add(token, cnt=0)
        # load embeddings
        self.embeddings = np.zeros([self.size(), self.embed_dim])
        for token in self.token2id.keys():
            if token in trained_embeddings:
                self.embeddings[self.get_id(token)] = trained_embeddings[token]

    def convert_to_ids(self, tokens, use_char_level=False):
        """
        Convert a list of tokens to ids, use unk_token if the token is not in vocab.
        Args:
            tokens: a list of token
        Returns:
            a list of ids
        """
        ids = [self.get_id(token) for token in tokens]
        char_ids = []
        if use_char_level:
            for token in tokens:
                chs = [self.get_char_id(ch) for ch in token]
                char_ids.append(chs)
        return ids, char_ids

    def recover_from_ids(self, ids,char_ids = [], stop_id=None):
        """
        Convert a list of ids to tokens, stop converting if the stop_id is encountered
        Args:
            ids: a list of ids to convert
            stop_id: the stop id, default is None
        Returns:
            a list of tokens
        """
        tokens = []
        for i in ids:
            tokens += [self.get_token(i)]
            if stop_id is not None and i == stop_id:
                break

        if len(char_ids) > 0:
            chars = []
            for i in char_ids:
                chars += [self.get_token(i)]
                if stop_id is not None and i == stop_id:
                    break
            return tokens , chars

        return tokens


