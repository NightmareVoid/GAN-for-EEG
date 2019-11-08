# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 21:29:14 2019
https://blog.csdn.net/qq_38826019/article/details/80786061 对生成网络损失的解释
@author: night
"""

from __future__ import print_function
 
import argparse
from random import randint
#import random
import os
import sys
import math
import tensorflow as tf
#import glbo
#import cv2
import scipy.io
import numpy as np
 
#from image_reader import *
from PIL import Image
from CGAN_tensorflowlz import *
from Vi_pattern import *
 
parser = argparse.ArgumentParser(description='')
 
parser.add_argument("--snapshot_dir", default='D:/EEG/model_gan', help="path of snapshots") #保存模型的路径
parser.add_argument("--rootpath", default='D:/EEG/model/AWD-LSTM/lz/50-250_9_vis/', help="path of vector") #条件向量路径
parser.add_argument("--vec_name", default='15631020158609197_ACC=0.88_loss=0.49_epoch=152_vis.mat', help="name of vector") #条件向量名
parser.add_argument("--hunhevec_name", default='15631020158609197_ACC=0.88_loss=0.49_epoch=152_hunhe_vis.mat', help="name of hunhevector") #条件向量名
parser.add_argument("--out_dir", default='D:/EEG/model_gan', help="path of train outputs") #训练时保存可视化输出的路径
parser.add_argument("--image_size", type=int, default=10, help="load image size") #网络输入的尺度
parser.add_argument("--random_seed", type=int, default=1234, help="random seed") #随机数种子
parser.add_argument('--base_lr', type=float, default=0.0002, help='initial learning rate for adam') #学习率
parser.add_argument('--epoch', dest='epoch', type=int, default=600, help='# of epoch')  #训练的epoch数量
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam') #adam优化器的beta1参数
parser.add_argument("--summary_pred_every", type=int, default=200, help="times to summary.") #训练中每过多少step保存训练日志(记录一下loss值)
parser.add_argument("--report_pred_every", type=int, default=50, help="times to write.") #训练中每过多少batch打印报告
parser.add_argument("--save_pred_every", type=int, default=3000, help="times to save.") #训练中每过多少step保存模型(可训练参数)
parser.add_argument("--lamda_l1_weight", type=float, default=1.0, help="L1 lamda") #训练中L1_Loss前的乘数
parser.add_argument("--lamda_grad_weight", type=float, default=1.0, help="L1 lamda") #训练中grad损失前的乘数
parser.add_argument("--lamda_gan_weight", type=float, default=1.0, help="GAN lamda") #训练中GAN_Loss前的乘数

parser.add_argument("--save_vic_pic", default='D:/EEG/model_gan/', help="path of training labels.") #可视化结果保存
 
args = parser.parse_args() #用来解析命令行参数
EPS = 1e-12 #EPS用于保证log函数里面的参数大于零
 
def save(saver, sess, logdir, step): #保存模型的save函数
      model_name = 'cDCWGAN-div_model' #保存的模型名前缀
      checkpoint_path = os.path.join(logdir, model_name) #模型的保存路径与名称
      if not os.path.exists(logdir): #如果路径不存在即创建
            os.makedirs(logdir)
      saver.save(sess, checkpoint_path, global_step=step) #保存模型
      print('The checkpoint has been created.')
 
def cv_inv_proc(img): #cv_inv_proc函数将读取图片时归一化的图片还原成原图
      img_rgb = (img + 1.) * 127.5
      return img_rgb.astype(np.float32) #返回bgr格式的图像，方便cv2写图像
 
def get_write_picture(picture, gen_label, label, height, width): #get_write_picture函数得到训练过程中的可视化结果
    picture_image = cv_inv_proc(picture) #还原输入的图像
    gen_label_image = cv_inv_proc(gen_label[0]) #还原生成的样本
    label_image = cv_inv_proc(label) #还原真实的样本(标签)
    inv_picture_image = cv2.resize(picture_image, (width, height)) #还原图像的尺寸
    inv_gen_label_image = cv2.resize(gen_label_image, (width, height)) #还原生成的样本的尺寸
    inv_label_image = cv2.resize(label_image, (width, height)) #还原真实的样本的尺寸
    output = np.concatenate((inv_picture_image, inv_gen_label_image, inv_label_image), axis=1) #把他们拼起来
    return output
def getdata(rootpath=args.rootpath,name=args.vec_name,hunhename=args.hunhevec_name):
      
      mat            = scipy.io.loadmat(rootpath+name)
      train_vec  = mat['vector'].astype('float32')
      train_tar  = mat['targets'].astype('int')
      train_vec  = train_vec[np.where(train_tar[0,:] != 7)]
      train_tar  = train_tar[0,np.where(train_tar[0,:] != 7)]
      train_vec  = train_vec[:,np.newaxis,np.newaxis,:]
      
      hunhemat            = scipy.io.loadmat(rootpath+hunhename)
      hunhetrain_vec  = hunhemat['vector'].astype('float32')
      hunhetrain_tar  = hunhemat['real_tar'].T.astype('int')
#      hunhetrain_vec  = hunhetrain_vec[np.where(train_tar[0,:] != 7)]
#      hunhetrain_tar  = hunhetrain_tar[0,np.where(train_tar[0,:] != 7)]
      hunhetrain_vec  = hunhetrain_vec[:,np.newaxis,np.newaxis,:]
      
      del(mat)
      del(hunhemat)

      true_pic,true_lab   = vi_pattern()
      return train_vec[0:32704],train_tar[:,0:32704],true_pic[0:7].astype('float32'),true_lab[0:7],hunhetrain_vec,hunhetrain_tar,true_pic[8:14]
'''#############################################################################'''
def l1_loss(src, dst): #定义l1_loss
    return tf.reduce_mean(tf.abs(src - dst))#对所有像素点取的均值
 
def main(): #训练程序的主函数
    if not os.path.exists(args.snapshot_dir): #如果保存模型参数的文件夹不存在则创建
        os.makedirs(args.snapshot_dir)
    if not os.path.exists(args.out_dir): #如果保存训练中可视化输出的文件夹不存在则创建
        os.makedirs(args.out_dir)
        
    train_vec,train_tar,true_pic,true_lab,hunhetrain_vec,hunhetrain_tar,hunhetrue_pic =  getdata()  #取数据
#    seq=np.random.randint(0,high=7,size=[64],dtype='int')
#    return train_vec,train_tar,true_pic,true_lab
    
    
    
    tf.set_random_seed(args.random_seed) #初始一下随机数
    
    
    
    #各个占位符   
    dis_true_train_pic = tf.placeholder(tf.float32,shape=[64, 10, 10, 1],name='dis_true_train_pic')#dic的真实数据训练数据
    dis_true_train_label = tf.placeholder(tf.float32,shape=[64, 1],name='dis_true_train_label')#dic的真实数据标签 0-6
    gen_train_vec = tf.placeholder(tf.float32,shape=[64, 1, 1, 64],name='gen_train_vec')#输入的训练向量  条件向量
    gen_train_noize = tf.placeholder(tf.float32,shape=[64, 1, 1, 64],name='gen_train_noize')
    gen_train_label = tf.placeholder(tf.float32,shape=[64, 1],name='gen_train_label') #输入的训练向量的标签 0-6
    gen_true_train_pic = tf.placeholder(tf.float32,shape=[64,10,10,1],name='gen_true_train_pic')#gen生成的图像对应的真实图像，用来计算像素级别l1损失
 
    
    '''生成器输出'''
    gen_pic = generator(image=gen_train_vec, noize=gen_train_noize,gf_dim=64, reuse=False, name='generator') #得到生成器的输出[-1,10,10,1]

    '''判别器输出'''
    dis_real = discriminator_lzcnn(image=dis_true_train_pic, df_dim=16, reuse=False, name="discriminator") #判别器返回的对真实标签的判别结果
    dis_fake = discriminator_lzcnn(image=gen_pic, df_dim=16, reuse=True, name="discriminator") #判别器返回的对生成(虚假的)标签判别结果

    
    '''生成器损失'''
#    gen_loss_GAN = tf.reduce_mean(-tf.log(dis_fake + EPS)) #计算生成器损失中的GAN_loss部分 这里它用的还是原始的损失函数
    gen_loss_GAN_negative = -tf.reduce_mean(dis_fake)#计算生成器损失中的GAN_loss部分，不加log  eps是在加log时才有用
    gen_loss_L1_negative = tf.reduce_mean(l1_loss(gen_pic, gen_true_train_pic)) #计算生成器损失中的L1_loss部分,其实是像素级别的损失函数
    gen_loss_negative = gen_loss_GAN_negative * args.lamda_gan_weight + gen_loss_L1_negative * args.lamda_l1_weight #计算生成器的loss
#    
#    gen_loss_GAN_positive = tf.reduce_mean(dis_fake)#计算生成器损失中的GAN_loss部分，不加log  eps是在加log时才有用
#    gen_loss_L1_positive = tf.reduce_mean(l1_loss(gen_pic, gen_true_train_pic)) #计算生成器损失中的L1_loss部分,其实是像素级别的损失函数
#    gen_loss_positive = gen_loss_GAN_positive * args.lamda_gan_weight + gen_loss_L1_positive * args.lamda_l1_weight #计算生成器的loss
 
    '''判别器损失'''
    dis_loss_negative = -tf.reduce_mean(dis_real) + tf.reduce_mean(dis_fake)#计算判别器的loss
#    dis_loss_positive = tf.reduce_mean(dis_real) - tf.reduce_mean(dis_fake)#计算判别器的loss
    
#    gen_loss_sum = tf.summary.scalar("gen_loss", gen_loss) #记录生成器loss的日志
#    dis_loss_sum = tf.summary.scalar("dis_loss", dis_loss) #记录判别器loss的日志
 
#    summary_writer = tf.summary.FileWriter(args.snapshot_dir, graph=tf.get_default_graph()) #日志记录器
    


 
#    '''用什么训练方法'''
#    # GP    差值的时候到底需不需要类别对应？同一种图片的时候没这问题
#    alpha = tf.random_uniform(shape=[64,1],minval=0.,maxval=1.)  
#    interpolates = alpha*gen_pic + (1-alpha)*dis_true_train_pic  
#    gradients = tf.gradients(discriminator_lzcnn(image=interpolates, df_dim=16, reuse=True, name="discriminator"),[interpolates])[0]  
#    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),reduction_indices=[1]))  
#    gradient_penalty = tf.reduce_mean((slopes-1.)**2)  
#    disc_cost += lamda * gradient_penalty  
#    
    # div
    p=6
    k=2
    gradients_real = tf.gradients(dis_real,[dis_true_train_pic])[0]
    gradients_fake = tf.gradients(dis_fake,[gen_pic])[0]
    gradients_real_norm = tf.reduce_sum(gradients_real**2, axis=[1, 2, 3])**(p / 2)
    gradients_fake_norm = tf.reduce_sum(gradients_fake**2, axis=[1, 2, 3])**(p / 2)
#    gradients_real_norm = tf.reduce_sum(gradients_real**2)**(p / 2)
#    gradients_fake_norm = tf.reduce_sum(gradients_fake**2)**(p / 2)
    grad_loss = tf.reduce_mean(gradients_real_norm + gradients_fake_norm) * k / 2
    dis_loss_negative_ori = dis_loss_negative
    dis_loss_negative += args.lamda_grad_weight * grad_loss
#    dis_loss_negative_with_grad =dis_loss_negative + 0.05*grad_loss
#    print('-------------gradients_fake.shape:',gradients_fake.shape,'-----------','\n') #64 10 10 1
#    print('-------------gradients_real_norm.shape:',gradients_real_norm,'-----------','\n')
#    print('-------------gradients_real_norm_1.shape:',gradients_real_norm_1.shape,'-----------','\n')
#    对每一个样本的梯度
    
    
    '''所有可训练参数'''
    g_vars = [v for v in tf.trainable_variables() if 'generator' in v.name] #所有生成器的可训练参数
    d_vars = [v for v in tf.trainable_variables() if 'discriminator' in v.name] #所有判别器的可训练参数    


    '''优化器'''
    gen_train_op = tf.train.AdamOptimizer(learning_rate=args.base_lr,beta1=0.5,beta2=0.9).minimize(gen_loss_negative,var_list=g_vars)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=args.base_lr,beta1=0.5,beta2=0.9).minimize(dis_loss_negative,var_list=d_vars)
    
##    d_optim = tf.train.AdamOptimizer(args.base_lr, beta1=args.beta1) #判别器训练器
##    g_optim = tf.train.AdamOptimizer(args.base_lr, beta1=args.beta1) #生成器训练器
## 
##    d_grads_and_vars = d_optim.compute_gradients(dis_loss, var_list=d_vars) #计算判别器参数梯度
##    d_train = d_optim.apply_gradients(d_grads_and_vars) #更新判别器参数
##    g_grads_and_vars = g_optim.compute_gradients(gen_loss, var_list=g_vars) #计算生成器参数梯度
##    g_train = g_optim.apply_gradients(g_grads_and_vars) #更新生成器参数
 
#    train_op = tf.group(disc_train_op, gen_train_op) #train_op表示了参数更新操作
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #设定显存不超量使用
    sess = tf.Session(config=config) #新建会话层
    init = tf.global_variables_initializer() #参数初始化器
 
    sess.run(init) #初始化所有可训练参数
 
#    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=50) #模型保存器
    gezhong={'GAN_negative_loss':[],'+L1_GAN_negative_loss':[],'dis_loss_negative':[],'grad_loss':[],
             'GAN_positive_loss':[],'+L1_GAN_positive_loss':[],'dis_loss_positivev':[]
             }
    '''训练步骤'''
    for epoch in range(args.epoch+1): #训练epoch数

        batch,i=0,0
#        gen_or_not = 0

        while i < train_tar.shape[1]: #每个训练epoch中的训练step数
            
            seq=np.random.randint(0,high=7,size=[64],dtype='int')
#            gen_or_not += 1
            gen_or_not = randint(1,5)
            true_pic_,true_lab_=true_pic[seq],true_lab[seq]  #用来训练dis的标准真实图片
            train_vec_ ,train_tar_ = train_vec[i:i+64],train_tar[0,i:i+64]  #用来训练gen的向量
            noize=np.random.normal(size=(64,1,1,64)).astype('float32')
            gen_true_train_pic_ = true_pic[train_tar_] #用来计算L1损失的假图片对应的真图片
       
            i+=64
            

#            feed_dict = { dis_true_train_pic : true_pic_, dis_true_train_label : true_lab_ ,
#                         gen_train_vec : train_vec_ , gen_train_label : train_tar_ ,
#                         gen_true_train_pic: gen_true_train_pic_
#                         } #构造feed_dict
            feed_dict = { dis_true_train_pic : true_pic_,
                         gen_train_vec : train_vec_ ,
                         gen_true_train_pic: gen_true_train_pic_,
                         gen_train_noize:noize
                         } #构造feed_dict
            
            gen_loss_GAN_negative_val, gen_loss_L1_negative_val,gen_loss_negative_val,dis_loss_negative_val,grad_loss_negative_val,dis_loss_negative_ori_val, _ = sess.run(
                        [gen_loss_GAN_negative, gen_loss_L1_negative,gen_loss_negative, dis_loss_negative,grad_loss,dis_loss_negative_ori,disc_train_op], feed_dict=feed_dict) #得到每个step中的生成器和判别器loss
            if gen_or_not == 1:
#                  gen_or_not = 0
                  gen_loss_GAN_negative_val, gen_loss_L1_negative_val,gen_loss_negative_val,dis_loss_negative_val,grad_loss_negative_val,dis_loss_negative_ori_val, _ = sess.run(
                              [gen_loss_GAN_negative, gen_loss_L1_negative,gen_loss_negative, dis_loss_negative,grad_loss,dis_loss_negative_ori,gen_train_op], feed_dict=feed_dict)
#            gen_loss_GAN_negative_val,gen_loss_negative_val, dis_loss_negative_val,grad_loss_negative_val,gen_loss_L1_negative_val,dis_loss_negative_with_grad_val = sess.run([gen_loss_GAN_negative,gen_loss_negative, dis_loss_negative,grad_loss,gen_loss_L1_negative,dis_loss_negative_with_grad], feed_dict=feed_dict)
#            gen_loss_GAN_positive_val,gen_loss_positive_val, dis_loss_positive_val,grad_loss_positive_val,gen_loss_L1_positive_val = sess.run([gen_loss_GAN_positive,gen_loss_positive, dis_loss_positive,grad_loss,gen_loss_L1_positive], feed_dict=feed_dict)
            
#            if counter % args.save_pred_every == 0: #每过save_pred_every次保存模型
#                save(saver, sess, args.snapshot_dir, counter)
#            if counter % args.summary_pred_every == 0: #每过summary_pred_every次保存训练日志
#                gen_loss_sum_value, discriminator_sum_value = sess.run([gen_loss_sum, dis_loss_sum], feed_dict=feed_dict)
#                summary_writer.add_summary(gen_loss_sum_value, counter)
#                summary_writer.add_summary(discriminator_sum_value, counter)
            if batch % args.report_pred_every == 0 and batch > 0:
#                print('------ epoch={:d} || batch={:d} \t gen_loss = {:.3f} || dis_loss = {:.3f}-----'.format(epoch, i, gen_loss_value, dis_loss_value))
                print('-------------------- epoch={:d} || batch={:d}-------------------'.format(epoch, i))
                print('------ 加负号 ：gen_loss = {:.4f}  || +L1_gen_loss = {:.4f}  || L1_loss = {:.4f} || dis_loss = {:.4f} || grad_loss={:.4f} || dis_loss_without_grad={:.4f} ---------\n'.format(
                            gen_loss_GAN_negative_val,gen_loss_negative_val,gen_loss_L1_negative_val, dis_loss_negative_val,grad_loss_negative_val,dis_loss_negative_ori_val))
#                print('------ 不加负 ：gen_loss = {:.3f}   || +L1_gen_loss = {:.3f}   || L1_loss = {:.3f} || dis_loss = {:.3f}  || grad_loss={:.3f} ---------'.format(gen_loss_GAN_positive_val,gen_loss_positive_val,gen_loss_L1_positive_val, dis_loss_positive_val,grad_loss_positive_val),'\n')
#                gezhong['GAN_negative_loss'].append(gen_loss_GAN_negative_val)
#                gezhong['+L1_GAN_negative_loss'].append(gen_loss_negative_val)
#                gezhong['dis_loss_negative'].append(dis_loss_negative_val)
#                gezhong['grad_loss'].append(grad_loss_negative_val)
#                gezhong['GAN_positive_loss'].append(gen_loss_GAN_positive_val)
#                gezhong['+L1_GAN_positive_loss'].append(gen_loss_positive_val)
#                gezhong['dis_loss_positivev'].append(dis_loss_positive_val)
            batch += 1
        if epoch % 30 == 0: #每过write_pred_every次写一下训练的可视化结果
            
            test_idx=randint(100,30000)

            train_vec_test ,train_tar_test = train_vec[test_idx:test_idx+64],train_tar[0,test_idx:test_idx+64]  #
            true_pic_test,true_lab_test=true_pic[train_tar_test],true_lab[train_tar_test]
            noize_test=np.random.normal(size=(64,1,1,64)).astype('float32')
            feed_dict_test = { gen_train_vec : train_vec_test,
                              gen_train_noize:noize_test
                         } #构造feed_dict
            vic_pic = sess.run(gen_pic, feed_dict=feed_dict_test) #run出生成器的输出
            
            
            vic_pic = vic_pic.reshape([640,10])
            true_pic_test=true_pic_test.reshape([640,10])
            expand_row=np.array([[i]*10 for i in range(10)]).reshape(-1)
            expand_column=np.array([[i]*10 for i in range(640)]).reshape(-1)
            vic_pic=vic_pic[expand_column,:]
            vic_pic=vic_pic[:,expand_row]
            true_pic_test=true_pic_test[expand_column,:]
            true_pic_test=true_pic_test[:,expand_row]
            vic_pic=np.append(vic_pic,true_pic_test,axis=1)
#            vic_pic = np.rollaxis(vic_pic.reshape([8,8,10,10]),2,start=1).reshape([80,80])
#            expand=np.array([[i]*10 for i in range(80)]).reshape(-1)
#            vic_pic=vic_pic[expand,:]
#            vic_pic=vic_pic[:,expand]
            
            img = Image.fromarray(vic_pic*255).convert('L')
#            print(train_tar_test,'\n')
#            print(true_lab_test,'\n')
            strr = '标签'
            for tar in train_tar_test:strr+=str(tar)
            img.save(args.save_vic_pic+'epoch={}_{}.jpg'.format(epoch,strr))
            
            #混合数据
            hunhe_idx=randint(100,450)
            hunhetrain_vec_test ,hunhetrain_tar_test = hunhetrain_vec[hunhe_idx:hunhe_idx+64],hunhetrain_tar[0,hunhe_idx:hunhe_idx+64]
            hunhetrue_pic_test=hunhetrue_pic[hunhetrain_tar_test-7]
            hunhefeed_dict_test = { gen_train_vec : hunhetrain_vec_test,
                              gen_train_noize:noize_test
                         } #构造feed_dict
            hunhevic_pic = sess.run(gen_pic, feed_dict=hunhefeed_dict_test)
            
            hunhevic_pic = hunhevic_pic.reshape([640,10])
            hunhetrue_pic_test=hunhetrue_pic_test.reshape([640,10])
            hunheexpand_row=np.array([[i]*10 for i in range(10)]).reshape(-1)
            hunheexpand_column=np.array([[i]*10 for i in range(640)]).reshape(-1)
            hunhevic_pic=hunhevic_pic[hunheexpand_column,:]
            hunhevic_pic=hunhevic_pic[:,hunheexpand_row]
            hunhetrue_pic_test=hunhetrue_pic_test[hunheexpand_column,:]
            hunhetrue_pic_test=hunhetrue_pic_test[:,hunheexpand_row]
            hunhevic_pic=np.append(hunhevic_pic,hunhetrue_pic_test,axis=1)
            
            hunheimg = Image.fromarray(hunhevic_pic*255).convert('L')
#            print(train_tar_test,'\n')
#            print(true_lab_test,'\n')
            strr = '混合标签'
            for tar in hunhetrain_tar_test:strr+=str(tar)
            hunheimg.save(args.save_vic_pic+'hunhe_'+'epoch={}_{}.jpg'.format(epoch,strr))

            
#    scipy.io.savemat('实验正负号到底有没有用.mat',gezhong) 
if __name__ == '__main__':
#    train_vec_,train_tar_,true_pic_,gen_true_train_pic_,seq=main()
#    train_vec,train_tar,true_pic,true_lab = main()
    main()
