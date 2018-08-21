# -*- coding:utf-8 -*-
import json
import tensorflow as tf

one = tf.ones(shape = [1,5] , dtype = tf.float32)
sample = tf.concat([one, 2 * one , 3 *one, 4 * one , 5 * one] ,0)
batch = tf.concat([sample , sample * 10 , sample * 100] , 0)
batch = tf.reshape(batch , [3 , 5 , 5])
Out = tf.nn.softmax(batch,-1)
Out2 = tf.matrix_set_diag(batch , tf.zeros(shape = [3, 5]))
Out3 = tf.shape(batch)[1]
ca = tf.cast(Out3,tf.int32)
print(type(ca))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(Out3))