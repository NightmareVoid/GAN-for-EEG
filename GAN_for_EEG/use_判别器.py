# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:45:19 2019

@author: night
"""

from __future__ import print_function
 
import argparse
from random import shuffle
import random
import os
import sys
import math
import tensorflow as tf
import numpy as np

from CGAN_tensorflowlz import *
from Vi_pattern import vi_pattern

#
#shizi               = np.array([[1,1,0,0,1,1],[1,1,0,0,1,1],[0,0,0,0,0,0],[0,0,0,0,0,0],[1,1,0,0,1,1],[1,1,0,0,1,1]])#十字
#zhongyang           = np.array([[1,1,1,1,1,1],[1,0,0,0,0,1],[1,0,0,0,0,1],[1,0,0,0,0,1],[1,0,0,0,0,1],[1,1,1,1,1,1]])#中央
#sizhou              = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,1,1,0,0],[0,0,1,1,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])#四周
#xiegang             = np.array([[0,0,0,1,1,1],[0,0,0,1,1,1],[1,0,0,0,1,1],[1,1,0,0,0,1],[1,1,1,0,0,0],[1,1,1,1,0,0]])#斜杠
#sidian              = np.array([[1,1,0,0,1,1],[1,1,0,0,1,1],[0,0,0,0,0,0],[0,0,0,0,0,0],[1,1,0,0,1,1],[1,1,0,0,1,1]])#周围4点
#shangxia            = np.array([[0,0,1,1,0,0],[0,0,1,1,0,0],[1,1,0,0,1,1],[1,1,0,0,1,1],[0,0,1,1,0,0],[0,0,1,1,0,0]])
#fanxie              = np.array([[1,1,1,0,0,0],[1,1,1,0,0,0],[0,1,1,1,0,0],[0,0,1,1,1,0],[0,0,0,1,1,1],[0,0,0,0,1,1]])
#space               = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
#image               = np.array([zhongyang,sizhou,shizi,xiegang,sidian,shangxia,fanxie,space])
image,_ = vi_pattern()
image=image[0:4].astype('float32')


#sequencetrain=np.random.randint(0,high=8,size=(640))
#sequenceval=np.random.randint(0,high=8,size=(128))
#trainx = image[sequencetrain]
#trainy = sequencetrain
#valx = image[sequenceval]
#valy = sequenceval

tf_x = tf.placeholder(tf.float32, [16, 10,10,1])
#tf_x = tf.reshape(tf_x,[-1,6,6,1])
tf_y = tf.placeholder(tf.int32, [16, 4])
outtf_y = tf.placeholder(tf.int32, [16, 4])
#
#
output,out_vec = discriminator_lzcnn_class(tf_x, reuse=False,name="discriminator")
#(image, targets, df_dim=64, reuse=False, name="discriminator")
loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

#acc = tf.metrics.accuracy(labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1))
#accuracy函数创建的是两个局部变量！！！ tf.metrics.accuracy 这个傻逼函数输出的是所有步数的平均精度和加上输入的这次的平均正确率，不是你输入的这次的！！！
acc = tf.equal(tf.argmax(tf_y, axis=1),tf.argmax(output, axis=1))
acc = tf.reduce_mean(tf.cast(acc, tf.float32))#将x的数据格式转化成dtype.例如，原来x的数据格式是bool， 那么将其转化成float以后，就能够将其转化成0和1的序列

sess= tf.Session()
#trainy = sess.run(tf.one_hot(trainy,8))
#valy = sess.run(tf.one_hot(valy,8))
sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))
#sess.run(tf.global_variables_initializer())
#sess.run(tf.local_variables_initializer())

valx = np.vstack((image,image,image,image))#
valy = sess.run(tf.one_hot(np.array([0,1,2,3]*4),4))

Loss=[]
Acc=[]
Val_out=[]
Val_out_acc=[]

for step in range(180):
    sequencetrain=np.random.randint(0,high=4,size=(16))
    trainx = image[sequencetrain]
    trainy = sequencetrain
    trainy = sess.run(tf.one_hot(trainy,4))
    
    loss_,_=sess.run([loss,train_op],{tf_x:trainx,tf_y:trainy})
    
    Loss.append(loss_)
#    
    if step % 10 == 0:
#        pass
        val_out = sess.run(output,{tf_x:valx})
        val_out_acc = sess.run(tf.argmax(val_out, axis=1))
        acc_out = sess.run(acc,{tf_x:valx,tf_y:valy})
        print('\n','(+_+)?  -------','%0.4f'% acc_out,'--------- ψ(｀∇´)ψ','\n')
#        print('\n','(+_+)?  -------','%0.4f'% acc_out[1],'--------- ψ(｀∇´)ψ','\n')
        Acc.append(acc_out)
        Val_out.append(val_out)
        Val_out_acc.append(val_out_acc)
_,vec=sess.run([output,out_vec],{tf_x:valx})

#    print('------%.4f------'% loss_)
    






