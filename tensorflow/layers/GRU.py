# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.rnn import RNNCell

from zoneout import ZoneoutWrapper

default_attn_size = 150
def bidirectional_GRU(inputs, inputs_len, cell=None, cell_fn=tf.contrib.rnn.GRUCell, units=default_attn_size, layers=1,
                      scope="Bidirectional_GRU", output=0, is_training=True, reuse=None):
    '''
    Bidirectional recurrent neural network with GRU cells.

    Args:
        inputs:     rnn input of shape (batch_size, timestep, dim)
        inputs_len: rnn input_len of shape (batch_size, )
        cell:       rnn cell of type RNN_Cell.
        output:     if 0, output returns rnn output for every timestep,
                    if 1, output returns concatenated state of backward and
                    forward rnn.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        shapes = inputs.get_shape().as_list()  # [batch_size , sequence_len_count_by_word , num_of_chars_in_cur_word]
        if len(shapes) > 3:  # char_level
            inputs = tf.reshape(inputs, (shapes[0] * shapes[1], shapes[2],
                                         -1))  # [batch数*句子中的单词数 ， 单词中的字符数 ，-1是char_embedding的大小] 单词中的字符数，len_word 有可能小于max_len_word
            inputs_len = tf.reshape(inputs_len, (shapes[0] * shapes[1],))

        # if no cells are provided, use standard GRU cell implementation
        if layers > 1:
            cell_fw = MultiRNNCell(
                [apply_dropout(cell_fn(units), size=inputs.shape[-1] if i == 0 else units, is_training=is_training)
                 for i in range(layers)])
            cell_bw = MultiRNNCell(
                [apply_dropout(cell_fn(units), size=inputs.shape[-1] if i == 0 else units, is_training=is_training)
                 for i in range(layers)])
        else:
            cell_fw, cell_bw = [apply_dropout(cell_fn(units), size=inputs.shape[-1], is_training=is_training) for _
                                in range(2)]

        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs,
                                                          sequence_length=inputs_len,
                                                          dtype=tf.float32)
        '''
        在前的是各个时间步的隐层状态，在后的是最后一个时间步的隐层状态
        一个（outputs, outputs_state）的一个元祖。其中，outputs=(outputs_fw, outputs_bw),是一个包含前向cell输出tensor和后向tensor输出tensor组成的元祖。
        若time_major=false，则两个tensor的shape为[batch_size, max_time, depth]，应用在文本中时，max_time可以为句子的长度（一般以最长的句子为准，短句需要做padding），depth为输入句子词向量的维度。
        最终的outputs需要使用tf.concat(outputs, 2)将两者合并起来。
        outputs_state = (outputs_state_fw， output_state_bw),包含了前向和后向最后的隐藏状态的组成的元祖。outputs_state_fw和output_state_bw的类型都是LSTMStateTuple。LSTMStateTuple由(c, h)组成，分别代表memory cell和hidden state
        '''

        if output == 0:  # 处理word_level的
            return tf.concat(outputs, 2)
        elif output == 1:  # 处理char embedding的时候，是走的这个分支
            return tf.reshape(tf.concat(states, 1),
                              (shapes[0], shapes[1], 2 * units))  # [batch, 句子中的单词数，双向的隐层的维度的]


def apply_dropout(inputs, size = None, is_training = True , whether_dropout = False , whether_zoneout = False):
    '''
    Implementation of Zoneout from https://arxiv.org/pdf/1606.01305.pdf
    '''
    if whether_dropout is False and whether_zoneout is False:
        return inputs
    if whether_zoneout is not False:
        return ZoneoutWrapper(inputs, state_zoneout_prob= whether_zoneout, is_training = is_training)
    elif is_training:
        return tf.contrib.rnn.DropoutWrapper(inputs,
                                            output_keep_prob = 1 - whether_dropout, # if it is constant and 1, no output dropout will be added.
                                            # variational_recurrent = True,
                                            # input_size = size,
                                            dtype = tf.float32)
    else:
        return inputs