import tensorflow as tf
import numpy as np
a = [1,2,3,4,5]
b = [a,a,a,a,a]
b = np.array(b,dtype=np.int32)
c = tf.constant(b , dtype = tf.float32)
c = tf.nn.softmax(c  , -1)
#d = tf.constant(np.ones([5,5],dtype = np.int32),dtype = tf.int32)
d = tf.constant(np.concatenate([np.ones([1,10]),np.ones([1,10]) * 2,np.ones([1,10]) * 3, np.ones([1,10]) * 4 , np.ones([1,10]) * 5] , axis = 0) , dtype = tf.float32)
e = tf.matmul(c , d , transpose_a=False , transpose_b = False)
with tf.Session():
    print("c = ",c.eval())
    print("d = ",d.eval())
    print("e = ",e.eval())


