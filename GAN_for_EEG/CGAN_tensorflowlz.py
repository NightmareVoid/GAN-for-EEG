# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 19:35:17 2019

@author: night
"""
#                        .::::.
#                      .::::::::.
#                     :::::::::::
#                  ..:::::::::::'
#               '::::::::::::'
#                 .::::::::::
#            '::::::::::::::..
#                 ..::::::::::::.
#               ``::::::::::::::::
#                ::::``:::::::::'        .:::.
#               ::::'   ':::::'       .::::::::.
#             .::::'      ::::     .:::::::'::::.
#            .:::'       :::::  .:::::::::' ':::::.
#           .::'        :::::.:::::::::'      ':::::.
#          .::'         ::::::::::::::'         ``::::.
#      ...:::           ::::::::::::'              ``::.
#     ```` ':.          ':::::::::'                  ::::..
#                        '.:::::'                    ':'````..
#                     美女保佑 永无BUG
import numpy as np
import tensorflow as tf
import math
#https://blog.csdn.net/tintinetmilou/article/details/81607721关于卷积输出的维度（通道）问题，以及卷积
#https://blog.csdn.net/songbinxu/article/details/85328522关于卷积核的维度问题
#https://blog.csdn.net/hustwayne/article/details/83989207 tensorflow计算转置卷积方式
#https://blog.csdn.net/qq_37691909/article/details/89490478
#构造可训练参数 tf1.0之前都需要先把要训练的权重定义下来
def make_var(name, shape, trainable = True):
    return tf.get_variable(name, shape, trainable = trainable)#获取一个已经存在的变量或者创建一个新的变量#此时没有指定initializer，默认为None 此时使用的是高斯初始化
 
#定义卷积层
def conv2d(input_, output_dim, kernel_size, stride, padding = "SAME", name = "conv2d", biased = False):
    input_dim = input_.get_shape()[-1]#通道数
    with tf.variable_scope(name):
        kernel = make_var(name = 'weights', shape=[kernel_size, kernel_size, input_dim, output_dim])
        output = tf.nn.conv2d(input_, kernel, [1, stride, stride, 1], padding = padding)#[filter_height, filter_width, in_channels, out_channels]
        if biased:
            biases = make_var(name = 'biases', shape = [output_dim])
            output = tf.nn.bias_add(output, biases)#1.0之前的偏置需要这样添加，就像C++
        return output
 
#定义空洞卷积层
def atrous_conv2d(input_, output_dim, kernel_size, dilation, padding = "SAME", name = "atrous_conv2d", biased = False):
    input_dim = input_.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = make_var(name = 'weights', shape = [kernel_size, kernel_size, input_dim, output_dim])
        output = tf.nn.atrous_conv2d(input_, kernel, dilation, padding = padding)
        if biased:
            biases = make_var(name = 'biases', shape = [output_dim])
            output = tf.nn.bias_add(output, biases)
        return output
 
#定义反卷积层
def deconv2d(input_, output_size,output_dim, kernel_size, stride, padding = "SAME", name = "deconv2d"):
    input_dim = input_.get_shape()[-1]
#    batch=input_.get_shape()[0]
#    input_height = int(input_.get_shape()[1])
#    input_width = int(input_.get_shape()[2])
    with tf.variable_scope(name):
        kernel = make_var(name = 'weights', shape = [kernel_size, kernel_size, output_dim, input_dim])
        output = tf.nn.conv2d_transpose(input_, kernel, [64, output_size, output_size, output_dim], [1, stride, stride, 1], padding =padding)#[filter_height, filter_width, out_channels, in_channels]
        return output
 
#定义batchnorm(批次归一化)层
def batch_norm(input_, name="batch_norm"):
    with tf.variable_scope(name):
        input_dim = input_.get_shape()[-1]
        scale = tf.get_variable("scale", [input_dim], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [input_dim], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input_, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input_-mean)*inv
        output = scale*normalized + offset
        return output
#定义线性层，用来处理条件向量
def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
                 initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias
 
#定义lrelu激活层
def lrelu(x, leak=0.2, name = "lrelu"):
    return tf.maximum(x, leak*x)

#全局平均池化
def avg_pool(x):
    return tf.nn.avg_pool(x,ksize=[1,x.get_shape()[1],x.get_shape()[2],1],strides=[1,x.get_shape()[1],x.get_shape()[2],1],padding='SAME')
 
#定义生成器，采用UNet架构，主要由8个卷积层和8个反卷积层组成
def generator(image,noize, gf_dim=64, reuse=False, name="generator"):
#    input_dim = int(image.get_shape()[-1]) #获取输入通道 64
    dropout_rate = 0.5 #定义dropout的比例
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
            
    # project the vector and reshape 最终得到3*3*64通道‘图’
#        image=linear(image,3*3*32,'G_linear',stddev=0.02, bias_start=0.0)
#        image=tf.reshape(image,[-1,3,3,32])

        image=tf.concat([noize, image], 3)
        d1 = deconv2d(input_=image,output_size=3, output_dim=32, kernel_size=3, stride=1, padding = "VALID",name='G_deConv_1')
        
#	#第一个卷积层，输出尺度[1, 128, 128, 64]
#        e1 = batch_norm(conv2d(input_=image, output_dim=gf_dim, kernel_size=4, stride=2, name='g_e1_conv'), name='g_bn_e1')
#	#第二个卷积层，输出尺度[1, 64, 64, 128]


	#第一个反卷积层，输出尺度[1, 2, 2, 512]
        d2 = deconv2d(input_=d1,output_size=5, output_dim=16, kernel_size=3, stride=2, name='G_deConv_2')
        d2 = tf.nn.dropout(d2, dropout_rate) #随机扔掉一般的输出
#        d1 = tf.concat([batch_norm(d1, name='G_bn_1'), e7], 3)
	#第二个反卷积层，输出尺度[1, 4, 4, 512]
        d3 = deconv2d(input_=tf.nn.relu(d2),output_size=10, output_dim=1, kernel_size=3, stride=2, name='G_deConv_3')
#        d2 = tf.nn.dropout(d2, dropout_rate) #随机扔掉一般的输出
#        d2 = tf.concat([batch_norm(d2, name='g_bn_d2'), e6], 3)

#	#第七个反卷积层，输出尺度[1, 128, 128, 64]
#        d7 = deconv2d(input_=tf.nn.relu(d6), output_dim=gf_dim, kernel_size=4, stride=2, name='g_d7')
#        d7 = tf.concat([batch_norm(d7, name='g_bn_d7'), e1], 3)
#	#第八个反卷积层，输出尺度[1, 256, 256, 3]
#        d8 = deconv2d(input_=tf.nn.relu(d7), output_dim=input_dim, kernel_size=4, stride=2, name='g_d8')
#        return tf.nn.tanh(d3)
        return tf.nn.tanh(d3) #对我的实验来说好像sigmod更方便一点，因为全部是0|1的
  
#定义生成器，采用UNet架构，主要由8个卷积层和8个反卷积层组成
def generator_with_linear(image, gf_dim=64, reuse=False, name="generator"):
#    input_dim = int(image.get_shape()[-1]) #获取输入通道 64
    dropout_rate = 0.5 #定义dropout的比例
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
            
    # project the vector and reshape 最终得到3*3*64通道‘图’
        image=linear(image,3*3*32,'G_linear',stddev=0.02, bias_start=0.0)
        image=tf.reshape(image,[-1,3,3,32])
        
#        d1 = deconv2d(input_=image,output_size=3, output_dim=32, kernel_size=3, stride=1, padding = "VALID",name='G_deConv_1')
        
#	#第一个卷积层，输出尺度[1, 128, 128, 64]
#        e1 = batch_norm(conv2d(input_=image, output_dim=gf_dim, kernel_size=4, stride=2, name='g_e1_conv'), name='g_bn_e1')
#	#第二个卷积层，输出尺度[1, 64, 64, 128]


	#第一个反卷积层，输出尺度[1, 2, 2, 512]
        d2 = deconv2d(input_=image,output_size=5, output_dim=16, kernel_size=3, stride=2, name='G_deConv_2')
        d2 = tf.nn.dropout(d2, dropout_rate) #随机扔掉一般的输出
#        d1 = tf.concat([batch_norm(d1, name='G_bn_1'), e7], 3)
	#第二个反卷积层，输出尺度[1, 4, 4, 512]
        d3 = deconv2d(input_=tf.nn.relu(d2),output_size=10, output_dim=1, kernel_size=3, stride=2, name='G_deConv_3')
#        d2 = tf.nn.dropout(d2, dropout_rate) #随机扔掉一般的输出
#        d2 = tf.concat([batch_norm(d2, name='g_bn_d2'), e6], 3)

#	#第七个反卷积层，输出尺度[1, 128, 128, 64]
#        d7 = deconv2d(input_=tf.nn.relu(d6), output_dim=gf_dim, kernel_size=4, stride=2, name='g_d7')
#        d7 = tf.concat([batch_norm(d7, name='g_bn_d7'), e1], 3)
#	#第八个反卷积层，输出尺度[1, 256, 256, 3]
#        d8 = deconv2d(input_=tf.nn.relu(d7), output_dim=input_dim, kernel_size=4, stride=2, name='g_d8')
#        return tf.nn.tanh(d3)
        return tf.nn.tanh(d3) #对我的实验来说好像sigmod更方便一点，因为全部是0|1的
 
#定义判别器
def discriminator_cnn(image, targets, df_dim=16, reuse=False, name="discriminator"):#全卷积方式
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
#        dis_input = tf.concat([image, targets], 3)
        dis_input = image
	#第1个卷积模块，输出尺度: 1*128*128*64
        h0 = lrelu(conv2d(input_ = dis_input, output_dim = df_dim, kernel_size = 4, stride = 1, name='d_h0_conv'))
        h0 = tf.nn.dropout(h0, 0.5)
	#第2个卷积模块，输出尺度: 1*64*64*128
        h1 = lrelu(conv2d(input_ = h0, output_dim = df_dim*2, kernel_size = 3, stride = 1, name='d_h1_conv'), name='lrelu1')
#	#第3个卷积模块，输出尺度: 1*32*32*256
#        h2 = lrelu(batch_norm(conv2d(input_ = h1, output_dim = df_dim*4, kernel_size = 4, stride = 2, name='d_h2_conv'), name='d_bn2'))
#	#第4个卷积模块，输出尺度: 1*32*32*512
#        h3 = lrelu(batch_norm(conv2d(input_ = h2, output_dim = df_dim*8, kernel_size = 4, stride = 1, name='d_h3_conv'), name='d_bn3'))
#	#最后一个卷积模块，输出尺度: 1*32*32*1
        dis_out = conv2d(input_ = h1, output_dim = 1, kernel_size = 3, stride = 1, name='d_h4_conv')
        
#        dis_out = tf.sigmoid(dis_out) #在输出之前经过sigmoid层，因为需要进行log运算
        return dis_out

#def discriminator_lzcnn(image, targets, df_dim=16, reuse=False, name="discriminator"):#卷积加全剧平均加最后全连接
def discriminator_lzcnn(image, df_dim=16, reuse=False, name="discriminator"):#卷积加全剧平均加最后全连接
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
#        dis_input = tf.concat([image, targets], 3)
        dis_input = image
	#第1个卷积模块，输出尺度: 1*128*128*64
        h0 = lrelu(conv2d(input_ = dis_input, output_dim = df_dim, kernel_size = 4, stride = 1, name='d_h0_conv'))
	#第2个卷积模块，输出尺度: 1*64*64*128
        h1 = lrelu(conv2d(input_ = h0, output_dim = df_dim*2, kernel_size = 3, stride = 1, name='d_h1_conv'), name='lrelu1')
#	#第3个卷积模块，输出尺度: 1*32*32*256
#        h2 = lrelu(conv2d(input_ = h1, output_dim = df_dim*4, kernel_size = 3, stride = 1, name='d_h2_conv'), name='d_bn2')
#	#第4个卷积模块，输出尺度: 1*32*32*512
#        h3 = lrelu(batch_norm(conv2d(input_ = h2, output_dim = df_dim*8, kernel_size = 4, stride = 1, name='d_h3_conv'), name='d_bn3'))
#	#最后一个卷积模块，输出尺度: 1*32*32*1
        output = lrelu(conv2d(input_ = h1, output_dim = df_dim*4, kernel_size = 3, stride = 1, name='d_h4_conv'))
#        output = lrelu(batch_norm(conv2d(input_ = h1, output_dim = df_dim*4, kernel_size = 3, stride = 1, name='d_h4_conv')))
        output = avg_pool(output)
        output = tf.reshape(output,[-1,1*1*(output.shape[3])])
        dis_out= tf.layers.dense(output,1)
        return dis_out
  
def discriminator_lzlinear(image, targets, df_dim=16, reuse=False, name="discriminator"):#使用全连接网络
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
#        dis_input = tf.concat([image, targets], 3)
        dis_input = tf.reshape(image,[-1,image.shape[1]*image.shape[2]*image.shape[3]])
        h0 = lrelu(tf.layers.dense(dis_input,64))
        h0 = tf.nn.dropout(h0, 0.5)
        h1 = lrelu(tf.layers.dense(h0,64))
        h1 = tf.nn.dropout(h1, 0.5)
        h2 = lrelu(tf.layers.dense(h1,64))

        dis_out= tf.layers.dense(h2,1)
        
#        dis_out = tf.sigmoid(dis_out) #在输出之前经过sigmoid层，因为需要进行log运算
        return dis_out
    
    
'''分类卷积网络'''
def discriminator_lzcnn_class(image, df_dim=16, reuse=False, name="discriminator"):#卷积加全剧平均加最后全连接
#def discriminator_lzcnn(image, df_dim=16, reuse=False, name="discriminator"):#卷积加全剧平均加最后全连接
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
#        dis_input = tf.concat([image, targets], 3)
        dis_input = image
	#第1个卷积模块，输出尺度: 64*
        h0 = lrelu(conv2d(input_ = dis_input, output_dim = df_dim, kernel_size = 4, stride = 1, name='d_h0_conv'))
	#第2个卷积模块，输出尺度: 1*64*64*128
        h1 = lrelu(conv2d(input_ = h0, output_dim = df_dim*2, kernel_size = 3, stride = 1, name='d_h1_conv'), name='lrelu1')
#	#第3个卷积模块，输出尺度: 1*32*32*256
#        h2 = lrelu(conv2d(input_ = h1, output_dim = df_dim*4, kernel_size = 3, stride = 1, name='d_h2_conv'), name='d_bn2')
#	#第4个卷积模块，输出尺度: 1*32*32*512
#        h3 = lrelu(batch_norm(conv2d(input_ = h2, output_dim = df_dim*8, kernel_size = 4, stride = 1, name='d_h3_conv'), name='d_bn3'))
#	#最后一个卷积模块，输出尺度: 1*32*32*1
        output = lrelu(conv2d(input_ = h1, output_dim = df_dim*4, kernel_size = 3, stride = 1, name='d_h4_conv'))
#        output = lrelu(batch_norm(conv2d(input_ = h1, output_dim = df_dim*4, kernel_size = 3, stride = 1, name='d_h4_conv')))
        output = avg_pool(output)
        output = tf.nn.tanh(tf.reshape(output,[-1,1*1*(output.shape[3])]))
        dis_out= tf.layers.dense(output,4)
        return dis_out,output