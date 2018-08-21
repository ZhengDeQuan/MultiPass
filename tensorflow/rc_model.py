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
This module implements the reading comprehension models based on:
1. the BiDAF algorithm described in https://arxiv.org/abs/1611.01603
2. the Match-LSTM algorithm described in https://openreview.net/pdf?id=B1-q5Pqxl
Note that we use Pointer Network for the decoding stage of both models.
"""

import os
import time
import logging
import json
import numpy as np
import tensorflow as tf
from utils import compute_bleu_rouge
from utils import normalize
from layers.basic_rnn import rnn
from layers.match_layer import MatchLSTMLayer
from layers.match_layer import AttentionFlowMatchLayer
from layers.pointer_net import PointerNetDecoder
from layers.self_attention_layer import Multi_Head_Att
from layers.GRU import bidirectional_GRU
import tensorflow.contrib as tc


#import allennlp
#from allennlp.commands.elmo import ElmoEmbedder
"""
DEFAULT_OPTIONS_FILE = "elmo_2x4096_512_2048cnn_2xhighway_options.json"
DEFAULT_WEIGHT_FILE = "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
A = ElmoEmbedder(
            options_file = "../../elmoConfig/"+DEFAULT_OPTIONS_FILE,
            weight_file = '../../elmoConfig/'+DEFAULT_WEIGHT_FILE,
            cuda_device = -1
)

"""
"""
def get_str_num(ids):
    '''
    ids is a list of length args.max_p_len
    '''
    return list(map(str, ids))
"""
"""
def get_allennlp_vec(batch_allennlp):
    batch_allennlp = A.embed_batch(batch_allennlp)  # batch_allennlp,List(numpy.ndarray)
    word_level = [sen[0] for sen in batch_allennlp]
    synt_level = [sen[1] for sen in batch_allennlp]
    semen_level = [sen[2] for sen in batch_allennlp]
    return np.concatenate([word_level, synt_level, semen_level], axis=-1)
"""

class RCModel(object):
    """
    Implements the main reading comprehension model.
    """

    def __init__(self, vocab, args):

        # logging
        self.logger = logging.getLogger("brc")

        # basic config
        self.algo = args.algo
        self.hidden_size = args.hidden_size
        self.char_hidden_size = args.char_hidden_size
        self.append_wordvec_size = args.append_wordvec_size
        self.optim_type = args.optim
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.use_dropout = args.dropout_keep_prob < 1

        # length limit
        self.max_p_num = args.max_p_num
        self.max_p_len = args.max_p_len
        self.max_q_len = args.max_q_len
        self.max_a_len = args.max_a_len
        self.max_word_len = args.max_word_len

        # the vocab
        self.vocab = vocab

        # session info
        sess_config = tf.ConfigProto(allow_soft_placement = True,
                                     log_device_placement = False)
        sess_config.gpu_options.allow_growth = True
        #sess_config.gpu_options.per_process_gpu_memory_fraction = 0.85
        self.sess = tf.Session(config=sess_config)

        self._build_graph()

        # save info
        self.saver = tf.train.Saver()

        # initialize the model
        self.sess.run(tf.global_variables_initializer())

    def _build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        start_t = time.time()
        self._setup_placeholders()
        self._embed()
        self._encode()
        self._match()
        self._self_attention()
        self._fuse()
        self._decode_boundary()
        self._decode_content()
        self._decode_verification()
        self._compute_boundary_loss()
        self._compute_content_loss()
        self._compute_verification_loss()
        self._compute_loss()
        self._create_train_op()
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))
        param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.all_params])
        self.logger.info('There are {} parameters in the model'.format(param_num))

    def _setup_placeholders(self):
        """
        Placeholders
        """
        #凡是以_char结尾的，都是char_level_embedding的
        self.p_char = tf.placeholder(tf.int32, [None, None, None])#batch_size, sequence_len , word_length
        self.q_char = tf.placeholder(tf.int32, [None, None, None])
        self.p_char_length = tf.placeholder(tf.int32 , [None , None])#batch_size , sequence_len
        self.q_char_length = tf.placeholder(tf.int32 , [None , None])


        self.p = tf.placeholder(tf.int32, [None, None])
        self.q = tf.placeholder(tf.int32, [None, None])
        self.p_allennlp = tf.placeholder(tf.float32 , [None , None , 3 * self.append_wordvec_size])
        self.q_allennlp = tf.placeholder(tf.float32 , [None , None , 3 * self.append_wordvec_size])#Input size (depth of inputs) must be accessible via shape inference, but saw value None
        self.p_length = tf.placeholder(tf.int32, [None])
        self.q_length = tf.placeholder(tf.int32, [None])
        self.start_label = tf.placeholder(tf.int32, [None])
        self.end_label = tf.placeholder(tf.int32, [None])
        self.real_pass = tf.placeholder(tf.int32, [None])
        self.sequence_label = tf.placeholder(tf.int32,[None,None]) #batch , 5 * sequence_len
        self.dropout_keep_prob = tf.placeholder(tf.float32)

    def _embed(self):
        """
        The embedding layer, question and passage share embeddings
        """
        with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
            self.word_embeddings = tf.get_variable(
                'word_embeddings',
                shape=(self.vocab.size(), self.vocab.embed_dim),
                initializer=tf.constant_initializer(self.vocab.embeddings),
                trainable=True
            )
            self.p_emb = tf.nn.embedding_lookup(self.word_embeddings, self.p)
            self.q_emb = tf.nn.embedding_lookup(self.word_embeddings, self.q)
            #self.p_emb = tf.concat([self.p_emb , self.p_allennlp],axis = -1)
            #self.q_emb = tf.concat([self.q_emb , self.q_allennlp],axis = -1)


            self.char_embeddings = tf.get_variable(
                'char_embeddings',
                shape = (self.vocab.char_size(), self.vocab.char_embed_dim),
                initializer=tf.constant_intializer(self.vocab.char_embeddings)
            )
            self.p_char_emb = tf.nn.embedding_lookup(self.char_embeddings , self.p_char)
            self.q_char_emb = tf.nn.embedding_lookup(self.char_embeddings , self.q_char)


    def _encode(self):
        """
        Employs two Bi-LSTMs to encode passage and question separately
        """
        with tf.variable_scope('passage_char_encoding'):
            shapes = self.p_char_emb.get_shape().as_list()#[batch_size , sequence_len_count_by_word , num_of_chars_in_cur_word , char_emb_dim]
            _ , self.sep_p_char_encodes = rnn('bi-gru' , inputs = tf.reshape(self.p_char_emb , (shapes[0] * shapes[1], shapes[2], -1)), length = tf.reshape(self.p_char_length , (shapes[0] * shapes[1] , )), hidden_size = self.char_hidden_size)
            #在前的是各个时间步的隐层状态，在后的是最后一个时间步的隐层状态
            self.sep_p_char_encodes = tf.reshape(self.sep_p_char_encodes , (shapes[0] , shapes[1] , -1))
        with tf.variable_scope('question_char_encoding'):
            shapes = self.q_char_emb.get_shape().as_list()
            _ , self.sep_q_char_encodes = rnn('bi-gru' , inputs = tf.reshape(self.p_char_emb , (shapes[0] * shapes[1], shapes[2], -1)), length = tf.reshape(self.q_char_length , (shapes[0] * shapes[1] , )), hidden_size = self.char_hidden_size)
            self.sep_q_char_encodes = tf.reshape(self.sep_q_char_encodes , (shapes[0] , shapes[1] , -1))

        self.p_emb = tf.concat([self.p_emb , self.sep_p_char_encodes] , axis = 2)
        self.q_emb = tf.concat([self.q_emb , self.sep_q_char_encodes] , axis = 2)

        with tf.variable_scope('passage_encoding'):
            self.sep_p_encodes, _ = rnn('bi-lstm', self.p_emb, self.p_length, self.hidden_size)
        with tf.variable_scope('question_encoding'):
            self.sep_q_encodes, _ = rnn('bi-lstm', self.q_emb, self.q_length, self.hidden_size)

        if self.use_dropout:
            self.sep_p_encodes = tf.nn.dropout(self.sep_p_encodes, self.dropout_keep_prob)
            self.sep_q_encodes = tf.nn.dropout(self.sep_q_encodes, self.dropout_keep_prob)

    def _match(self):
        """
        The core of RC model, get the question-aware passage encoding with either BIDAF or MLSTM
        """
        if self.algo == 'MLSTM':
            match_layer = MatchLSTMLayer(self.hidden_size)
        elif self.algo == 'BIDAF':
            match_layer = AttentionFlowMatchLayer(self.hidden_size)
        else:
            raise NotImplementedError('The algorithm {} is not implemented.'.format(self.algo))
        self.match_p_encodes, _ = match_layer.match(self.sep_p_encodes, self.sep_q_encodes,
                                                    self.p_length, self.q_length)
        if self.use_dropout:
            self.match_p_encodes = tf.nn.dropout(self.match_p_encodes, self.dropout_keep_prob)

    def _self_attention(self):
        self.match_p_self_attention = Multi_Head_Att(Q = self.match_p_encodes, K = self.match_p_encodes, V = self.match_p_encodes, nb_head = 8, size_per_head = self.hidden_size, Q_len=self.p_length, V_len=self.p_length)
        #[batch, sequence_lenQ, nb_head * size_per_head]

    def _fuse(self):
        """
        Employs Bi-LSTM again to fuse the context information after match layer
        """
        with tf.variable_scope('fusion'):
            # self.fuse_p_encodes, _ = rnn('bi-lstm', self.match_p_encodes, self.p_length,
            #                              self.hidden_size, layer_num=1)
            self.fuse_p_encodes, _ = rnn('bi-lstm', self.match_p_self_attention, self.p_length,
                                         self.hidden_size, layer_num=1)
            if self.use_dropout:
                self.fuse_p_encodes = tf.nn.dropout(self.fuse_p_encodes, self.dropout_keep_prob)

    def _decode_boundary(self):
        """
        Employs Pointer Network to get the the probs of each position
        to be the start or end of the predicted answer.
        Note that we concat the fuse_p_encodes for the passages in the same document.
        And since the encodes of queries in the same document is same, we select the first one.
        """
        with tf.variable_scope('Answer_boundary_prediction_layer'):
            batch_size = tf.shape(self.start_label)[0]
            concat_passage_encodes = tf.reshape(
                self.fuse_p_encodes,
                [batch_size, -1, 2 * self.hidden_size]
            )
            no_dup_question_encodes = tf.reshape(
                self.sep_q_encodes,
                [batch_size, -1, tf.shape(self.sep_q_encodes)[1], 2 * self.hidden_size]
            )[0:, 0, 0:, 0:]
        decoder = PointerNetDecoder(self.hidden_size)
        self.start_probs, self.end_probs = decoder.decode(concat_passage_encodes,
                                                          no_dup_question_encodes)

    def _compute_boundary_loss(self):
        def sparse_nll_loss(probs, labels, epsilon=1e-9, scope=None):
            """
            negative log likelyhood loss
            """
            with tf.name_scope(scope, "log_loss"):
                labels = tf.one_hot(labels, tf.shape(probs)[1], axis=1)
                losses = - tf.reduce_sum(labels * tf.log(probs + epsilon), 1)
            return losses

        with tf.variable_scope("boundary_loss"):
            self.start_loss = sparse_nll_loss(probs=self.start_probs, labels=self.start_label)
            self.end_loss = sparse_nll_loss(probs=self.end_probs, labels=self.end_label)
            self.boundary_loss = tf.reduce_mean(tf.add(self.start_loss, self.end_loss))
        # self.all_params = tf.trainable_variables()
        # if self.weight_decay > 0:
        #     with tf.variable_scope('l2_loss'):
        #         l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
        #     self.loss += self.weight_decay * l2_loss



    def _decode_content(self):
        with tf.variable_scope('Answer_content_modeling_layer'):

            batch_size = tf.shape(self.start_label)[0]
            concat_passage_encodes = tf.reshape(
                self.fuse_p_encodes,
                [batch_size, -1, 2 * self.hidden_size]
            )
            #[batch_size  , 5 * p , 2 * self.hidden_size]
   
            logits = tc.layers.fully_connected(
                tc.layers.fully_connected(concat_passage_encodes , num_outputs= self.hidden_size , activation_fn=tf.nn.relu),
                num_outputs = 2,
                activation_fn=None
            )
            #[batch,5 * p,2] 完成w1T ReLU（W2v{passage} ）的操作
            self.content_probs = tf.nn.softmax(logits , -1)#[batch, 5 * p , 2]在 2，这个维度上归一化 ， 评估每一个位置的为正确答案的概率
            self.yes_probs = self.content_probs[0:,0:,1]#[batch , 5 * p]
            self.yes_probs = tf.expand_dims(self.yes_probs , 2)

    def _compute_content_loss(self):
        def sparse_null_loss(probs , labels , epsilon=1e-9 , scope = None):
            with tf.name_scope(scope,"log_loss"):
                labels = tf.one_hot(labels, depth=2)
                losses = -tf.reduce_sum(labels * tf.log(probs + epsilon) , 2)
                loss = tf.reduce_mean(losses) #在每个sample和每个sample的每个位置上，进行mean操作
            return loss

        with tf.variable_scope('Content_loss'):
            self.content_loss = sparse_null_loss(self.content_probs, self.sequence_label)
            """
            '''
            用两层tf.while_loop()循环将只有start，end label的情况，转换成长度为5 * p的向量，向量的某一个index = 1则这个index对应的passage是答案的一部分，是0，则vice verse
            '''
            start_ta = tf.TensorArray(dtype=tf.int32, size=batch_size)
            end_ta = tf.TensorArray(dtype=tf.int32, size=batch_size)
            # 一个单元数为max_time的Array，其中的每个元素都是Tensor，Tensor的值和shape都还没有确定，
            # tensor内部不支持循环，比如一个m*n的矩阵A，不可以 for （i = 0 ； i < m : i ++） A[i].oper() 但是由Tensor组成的数组可以这么做
            i = tf.constant(0)
            start_ta = start_ta.unstack(self.start_label)
            end_ta = end_ta.unstack(self.end_label)
            emit_ta = tf.TensorArray(dtype=tf.int32, size=batch_size)

            def loop_fn2_get_label_seque(t, start_index, end_index, emit_ta_inner):
                def true_fn():
                    return 1

                def false_fn():
                    return 0

                res = tf.cond(tf.logical_and(tf.greater_equal(t, start_index), tf.less_equal(t, end_index)),
                              true_fn=true_fn, false_fn=false_fn)
                emit_ta_inner = emit_ta_inner.write(t, res)
                return [t + 1, start_index, end_index, emit_ta_inner]

            def loop_fn1_get_label_batch(i, emit_ta):  # 外层循环 batch 级别的循环
                start_ta_seq = start_ta.read(i)
                end_ta_seq = end_ta.read(i)

                t = tf.constant(0)
                sequence_len = tf.shape(self.yes_probs)[1]
                emit_ta_inner = tf.TensorArray(dtype=tf.int32, size=sequence_len)
                _1, _2, _3, res = tf.while_loop(
                    cond=lambda t, _1, _2, _3: t < sequence_len,
                    body=loop_fn2_get_label_seque,
                    loop_vars=[t, start_ta_seq, end_ta_seq, emit_ta_inner]
                )
                res = res.stack()
                emit_ta = emit_ta.write(i, res)
                return [i + 1, emit_ta]

            _1, res = tf.while_loop(
                cond=lambda i, _1: i < batch_size,
                body=loop_fn1_get_label_batch,
                loop_vars=[i, emit_ta]
            )
            res = res.stack()
            self.labels = tf.to_float(res)
            self.content_loss = sparse_null_loss(content_probs, self.labels)
            """


    def _decode_verification(self):
        with tf.variable_scope("Cross_passage_verification"):
            batch_size = tf.shape(self.start_label)[0]
            content_probs = tf.reshape(self.yes_probs , [tf.shape(self.p_emb)[0] , tf.shape(self.p_emb)[1], 1]) # [batch * 5 , p , 1]
            #content_probs = tf.nn.softmax(content_probs , 1) #原来的公式中是用加和来归一化的
            ver_P = content_probs * self.p_emb
            #[batch_size , -1 , tf.shape(self.p_emb)[1] , self.p_meb.get_shape().as_list()[2] ]
            #ver_P = tf.reshape(ver_P , [batch_size , -1 , tf.shape(self.p_emb)[1] , 3 * self.append_wordvec_size + self.vocab.embed_dim ]) #[batch , 5 , p , wordvec dimension = 3 * 1024 + 300]
            ver_P = tf.reshape(ver_P , [batch_size , -1 , tf.shape(self.p_emb)[1] , self.vocab.embed_dim ]) #[batch , 5 , seq_len , emb_dim]
            RA = tf.reduce_mean(ver_P , axis=2) # [batch , 5 , wordvec]
            #如果上面这句话#content_probs = tf.nn.softmax(content_probs , 1) #原来的公式中是用加和来归一化的，
            #没有被注释掉，就是reduce_sum
            #print("RA_concated.shape = ",RA.shape)
            #Given the representation of the answer candidates from all passages {rAi }, each answer candidate then attends to other candidates to collect supportive information via attention mechanism
            #tf.batch_mat_mul()
            S = tf.matmul(RA , RA , transpose_a = False , transpose_b = True)# [batch , 5 , 5]
            S = tf.matrix_set_diag(input = S , diagonal = tf.zeros(shape = [batch_size ,  tf.shape(S)[1] ] ,dtype = S.dtype)) #[batch , 5 , 5] the main digonal of innermost matrices is all 0
            S = tf.nn.softmax(S , -1) #[batch , 5 , 5] 每一行都是归一化过的了
            RA_Complementary = tf.matmul(S , RA , transpose_a = False , transpose_b = False)
            #Here ̃rAi is the collected verification information from other passages based on the attention weights. Then we pass it together with the original representation rAi to a fully connected layer
            RA_concated = tf.concat([RA ,
                                     RA_Complementary,
                                     RA * RA_Complementary] , -1) # [batch , 5 , 3 * (3 * 1024 + 300) = 10116]
                                                                  # [batch , 5 , 3 * 300] 在没有加elmo向量的时候是这样的
            
            g = tc.layers.fully_connected(RA_concated , num_outputs= 1, activation_fn=None)#[batch , 5 , 1]
            g = tf.reshape(g, shape= [batch_size , -1]) #[batch , 5 ]
            self.pred_pass_prob = tf.nn.softmax(g , - 1)#[batch , 5 ]

    def _compute_verification_loss(self):
        def sparse_nll_loss(probs, labels, epsilon=1e-9, scope=None):
            """
            negative log likelyhood loss
            """
            with tf.name_scope(scope, "log_loss"):
                labels = tf.one_hot(labels, tf.shape(probs)[1], axis=1)
                losses = - tf.reduce_mean(tf.reduce_sum(labels * tf.log(probs + epsilon), 1)) #在每个样本的各个位置上要加和， 在所有的样本上要平均
            return losses
        with tf.variable_scope('verification_loss'):
            self.verification_loss = sparse_nll_loss(self.pred_pass_prob, self.real_pass)


    def _compute_loss(self,b1=1.0,b2=0.1,b3=0.225):
        #total = 1.0 * (b1  + b2  + b3)
        #b1 = 1.0 * b1 / total
        #b2 = 1.0 * b2 / total
        #b3 = 1.0 * b3 / total
        with tf.variable_scope("joint_loss"):
            self.loss = b1 * self.boundary_loss + b2 * self.content_loss + b3 * self.verification_loss
            self.all_params = tf.trainable_variables()
            if self.weight_decay > 0:
                with tf.variable_scope('l2_loss'):
                    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
                self.loss += self.weight_decay * l2_loss


    def _create_train_op(self):
        """
        Selects the training algorithm and creates a train operation with it
        """
        if self.optim_type == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optim_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optim_type == 'rprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optim_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))
        self.train_op = self.optimizer.minimize(self.loss)

    def _train_epoch(self, train_batches, dropout_keep_prob):
        """
        Trains the model for a single epoch.
        Args:
            train_batches: iterable batch data for training
            dropout_keep_prob: float value indicating dropout keep probability
        """
        total_num, total_loss = 0, 0
        log_every_n_batch, n_batch_loss = 50, 0
        for bitx, batch in enumerate(train_batches, 1):
            #p_allennlpd = [get_str_num(ids) for ids in batch['passage_token_ids']]
            #q_allennlpd = [get_str_num(ids) for ids in batch['question_token_ids']]
            #p_allennlpd = get_allennlp_vec(p_allennlpd)
            #q_allennlpd = get_allennlp_vec(q_allennlpd) # shape[batch,sequence_len , 3 * 1024 = 3072]
            #print(p_allennlpd.shape) # [32 * 5 = 160 , 500 , 3 * 1024 = 3072]
            #print(q_allennlpd.shape) # [32 * 5 , 500 , 3 * 1024 = 3072]
        
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         #self.p_allennlp : p_allennlpd,
                         #self.q_allennlp : q_allennlpd,
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],

                         self.p_char : batch['passage_char_ids'],
                         self.q_char : batch['question_char_ids'],
                         self.p_char_length:batch['passage_char_length'],
                         self.q_char_length:batch['question_char_length'],

                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         self.real_pass:batch['real_pass'],
                         self.sequence_label:batch['sequence_label'],
                         self.dropout_keep_prob: dropout_keep_prob}
            #_, loss = self.sess.run([self.train_op, self.loss], feed_dict)
            _, loss, boundary_loss , content_loss , verification_loss = self.sess.run([self.train_op, self.loss, self.boundary_loss , self.content_loss , self.verification_loss], feed_dict)
            # print("macth = ",match_p_self_attention)
            # exit(789)
            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])
            n_batch_loss += loss
            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                self.logger.info('Average loss from batch {} to {} is {} , bloss is {} , closs is {} , vloss is {}'.format(
                    bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch ,boundary_loss , content_loss , verification_loss))
                n_batch_loss = 0
        return 1.0 * total_loss / total_num

    def train(self, data, epochs, batch_size, save_dir, save_prefix,
              dropout_keep_prob=1.0, evaluate=True):
        """
        Train the model with data
        Args:
            data: the BRCDataset class implemented in dataset.py
            epochs: number of training epochs
            batch_size:
            save_dir: the directory to save the model
            save_prefix: the prefix indicating the model type
            dropout_keep_prob: float value indicating dropout keep probability
            evaluate: whether to evaluate the model on test set after each epoch
        """
        pad_id = self.vocab.get_id(self.vocab.pad_token)
        max_bleu_4 = 0
        for epoch in range(1, epochs + 1):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            train_batches = data.gen_mini_batches('train', batch_size, pad_id, shuffle=True)
            train_loss = self._train_epoch(train_batches, dropout_keep_prob)
            self.logger.info('Average train loss for epoch {} is {}'.format(epoch, train_loss))

            if evaluate:
                self.logger.info('Evaluating the model after epoch {}'.format(epoch))
                if data.dev_set is not None:
                    eval_batches = data.gen_mini_batches('dev', batch_size, pad_id, shuffle=False)
                    eval_loss, bleu_rouge = self.evaluate(eval_batches)
                    self.logger.info('Dev eval loss {}'.format(eval_loss))
                    self.logger.info('Dev eval result: {}'.format(bleu_rouge))

                    if bleu_rouge['Bleu-4'] > max_bleu_4:
                        self.save(save_dir, save_prefix)
                        max_bleu_4 = bleu_rouge['Bleu-4']
                else:
                    self.logger.warning('No dev set is loaded for evaluation in the dataset!')
            else:
                self.save(save_dir, save_prefix + '_' + str(epoch))

    def evaluate(self, eval_batches, result_dir=None, result_prefix=None, save_full_info=False):
        """
        Evaluates the model performance on eval_batches and results are saved if specified
        Args:
            eval_batches: iterable batch data
            result_dir: directory to save predicted answers, answers will not be saved if None
            result_prefix: prefix of the file for saving predicted answers,
                           answers will not be saved if None
            save_full_info: if True, the pred_answers will be added to raw sample and saved
        """
        pred_answers, ref_answers = [], []
        total_loss, total_num = 0, 0
        for b_itx, batch in enumerate(eval_batches):
            #p_allennlpd = [get_str_num(ids) for ids in batch['passage_token_ids']]
            #q_allennlpd = [get_str_num(ids) for ids in batch['question_token_ids']]
            #p_allennlpd = get_allennlp_vec(p_allennlpd)
            #q_allennlpd = get_allennlp_vec(q_allennlpd)  # shape[batch,sequence_len , 3 * 1024 = 3072]
            # print(p_allennlpd.shape) # [32 * 5 = 160 , 500 , 3 * 1024 = 3072]
            # print(q_allennlpd.shape) # [32 * 5 , 500 , 3 * 1024 = 3072]

            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         #self.p_allennlp: p_allennlpd,
                         #self.q_allennlp: q_allennlpd,
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],

                         self.p_char: batch['passage_char_ids'],
                         self.q_char: batch['question_char_ids'],
                         self.p_char_length: batch['passage_char_length'],
                         self.q_char_length: batch['question_char_length'],

                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         self.real_pass: batch['real_pass'],
                         self.sequence_label:batch['sequence_label'],
                         self.dropout_keep_prob: 1.0}
            start_probs, end_probs, loss = self.sess.run([self.start_probs,
                                                          self.end_probs, self.loss], feed_dict)

            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])

            padded_p_len = len(batch['passage_token_ids'][0])
            for sample, start_prob, end_prob in zip(batch['raw_data'], start_probs, end_probs):

                best_answer = self.find_best_answer(sample, start_prob, end_prob, padded_p_len)
                if save_full_info:
                    sample['pred_answers'] = [best_answer]
                    pred_answers.append(sample)
                else:
                    pred_answers.append({'question_id': sample['question_id'],
                                         'question_type': sample['question_type'],
                                         'answers': [best_answer],
                                         'entity_answers': [[]],
                                         'yesno_answers': []})
                if 'answers' in sample:
                    ref_answers.append({'question_id': sample['question_id'],
                                         'question_type': sample['question_type'],
                                         'answers': sample['answers'],
                                         'entity_answers': [[]],
                                         'yesno_answers': []})

        if result_dir is not None and result_prefix is not None:
            result_file = os.path.join(result_dir, result_prefix + '.json')
            with open(result_file, 'w') as fout:
                for pred_answer in pred_answers:
                    fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')

            self.logger.info('Saving {} results to {}'.format(result_prefix, result_file))

        # this average loss is invalid on test set, since we don't have true start_id and end_id
        ave_loss = 1.0 * total_loss / total_num
        # compute the bleu and rouge scores if reference answers is provided
        if len(ref_answers) > 0:
            pred_dict, ref_dict = {}, {}
            for pred, ref in zip(pred_answers, ref_answers):
                question_id = ref['question_id']
                if len(ref['answers']) > 0:
                    pred_dict[question_id] = normalize(pred['answers'])
                    ref_dict[question_id] = normalize(ref['answers'])
            bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
        else:
            bleu_rouge = None
        return ave_loss, bleu_rouge

    def find_best_answer(self, sample, start_prob, end_prob, padded_p_len):
        """
        Finds the best answer for a sample given start_prob and end_prob for each position.
        This will call find_best_answer_for_passage because there are multiple passages in a sample
        """
        best_p_idx, best_span, best_score = None, None, 0
        for p_idx, passage in enumerate(sample['passages']):
            if p_idx >= self.max_p_num:
                continue
            passage_len = min(self.max_p_len, len(passage['passage_tokens']))
            answer_span, score = self.find_best_answer_for_passage(
                start_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                end_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                passage_len)
            if score > best_score:
                best_score = score
                best_p_idx = p_idx
                best_span = answer_span
        if best_p_idx is None or best_span is None:
            best_answer = ''
        else:
            best_answer = ''.join(
                sample['passages'][best_p_idx]['passage_tokens'][best_span[0]: best_span[1] + 1])
        return best_answer

    def find_best_answer_for_passage(self, start_probs, end_probs, passage_len=None):
        """
        Finds the best answer with the maximum start_prob * end_prob from a single passage
        """
        if passage_len is None:
            passage_len = len(start_probs)
        else:
            passage_len = min(len(start_probs), passage_len)
        best_start, best_end, max_prob = -1, -1, 0
        for start_idx in range(passage_len):
            for ans_len in range(self.max_a_len):
                end_idx = start_idx + ans_len
                if end_idx >= passage_len:
                    continue
                prob = start_probs[start_idx] * end_probs[end_idx]
                if prob > max_prob:
                    best_start = start_idx
                    best_end = end_idx
                    max_prob = prob
        return (best_start, best_end), max_prob

    def save(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model saved in {}, with prefix {}.'.format(model_dir, model_prefix))

    def restore(self, model_dir, model_prefix):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model restored from {}, with prefix {}'.format(model_dir, model_prefix))
