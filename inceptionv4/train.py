# -*- coding: utf-8 -*-
import tensorflow as tf
import preprocessing
from cnn import convolution2d, max_pool, algorithm, affine, batch_norm_layer, gap
import inception_v4
import data
import numpy as np
import utils
from inception_v4 import stem, stem_1, stem_2, reductionA, reductionB, blockA, blockB, blockC
import cam
import aug
import random
import argparse
import os
import time
from tensorflow.examples.tutorials.mnist import input_data
import sys

def show_progress(i,max_iter):
    msg='\r progress {}/{}'.format(i, max_iter)
    sys.stdout.write(msg)
    sys.stdout.flush()
def input_pipeline(type='mnist'):

    mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)
    train_imgs=mnist.train.images.reshape([-1 , 28,28,1])
    train_labs=mnist.train.labels
    val_imgs = mnist.test.images.reshape([-1, 28, 28, 1])
    val_labs = mnist.test.labels
    train_imgs, train_labs, test_imgs, test_labs = preprocessing.get_cifar_data()
    return (train_imgs , train_labs , val_imgs , val_labs)
def train(max_iter ,learning_rate , check_point, optimizer='AdamOptimizer',restored_model_folder_path=None , restored_path_folder_path=None):
    ##########################setting############################
    train_imgs , train_labs, val_imgs , val_labs=input_pipeline()



    n,image_height,image_width,image_color_ch=np.shape(train_imgs)
    n_classes=len(train_labs[0])


    x_ = tf.placeholder(dtype=tf.float32, shape=[None, image_height, image_width, image_color_ch], name='x_')
    y_ = tf.placeholder(dtype=tf.int32, shape=[None, n_classes], name='y_')
    phase_train = tf.placeholder(dtype=tf.bool, name='phase_train')
    ##########################structure##########################
    top_conv=inception_v4.structure_C(x_)
    y_conv = gap('gap', top_conv, n_classes)
    cam_ = cam.get_class_map('gap', top_conv, 0, image_height)
    #################fully connected#############################
    """
    layer=tf.contrib.layers.flatten(layer)
    print layer.get_shape()
    layer = affine('fully_connect', layer, 1024 ,keep_prob=0.5)
    y_conv=affine('end_layer' , layer , n_classes , keep_prob=1.0)
    """
    # cam = get_class_map('gap', top_conv, 0, im_width=image_width)
    pred, pred_cls, cost, train_op, correct_pred, accuracy = algorithm(y_conv, y_, learning_rate , optimizer)
    saver = tf.train.Saver()
    config = tf.ConfigProto(
        device_count={'GPU': 1},
        log_device_placement=False
    )
    #config.gpu_options.allow_growth = True

    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    ########################training##############################
    max_val = 0;
    train_acc = 0;
    train_loss = 0;
    #next_batch


    start_time=time.time()
    try:
        f=open('log.txt' ,'w')
        for step in range(max_iter):
            show_progress(step, max_iter)

            if step % check_point == 0:
                end_time=time.time()
                # cam.inspect_cam(sess, cam_ , top_conv,test_imgs, test_labs, step , 50 , x_,y_ , y_conv  )
                val_acc, val_loss = sess.run([accuracy, cost],
                                             feed_dict={x_: val_imgs, y_: val_labs, phase_train: False})
                utils.write_acc_loss(f, train_acc, train_loss, val_acc, val_loss)
                print 'valicataion accuracy , and loss \n', val_acc, val_loss
                """
                if val_acc > max_val:
                    saver.save(sess, restored_model_folder_path + '/best_acc.ckpt')
                    print 'model was saved!'
                    max_val = val_acc
                """
                print 'time was consumed : ',end_time- start_time
                start_time=time.time()

            # names = ['cataract', 'glaucoma', 'retina', 'retina_glaucoma','retina_cataract', 'cataract_glaucoma', 'normal']
            batch_xs, batch_ys = data.next_batch(train_imgs, train_labs , batch_size=120)
            #batch_xs = aug.aug_level_1(batch_xs)
            #utils.np2images(batch_xs , './debug')
            train_acc, train_loss, _ = sess.run([accuracy, cost, train_op],
                                                feed_dict={x_: batch_xs, y_: batch_ys, phase_train: True})
            f.flush()
        f.close()
    except Exception as e :
        print e
    #utils.draw_grpah(log_saved_file_path , graph_saved_folder_path , check_point)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iter", help='iteration',type=int)
    parser.add_argument("--batch_size" ,help='batch size ' ,type=int)
    parser.add_argument("--learning_rate" , help='learning rate ',type=float)
    parser.add_argument("--structure" , help = 'what structrue you need')
    parser.add_argument("--gpu",help='used gpu')
    parser.add_argument("--check_point" , help='' , type=int)
    parser.add_argument("--optimizer",help='')
    args = parser.parse_args()

    #train_with_redfree(args.iter , args.batch_size , args.learning_rate , args.structure , restored_model_folder_path=None)
    #train_with_specified_gpu(gpu_device='/gpu:1')
    restored_model_folder_path='./cnn_model/fundus/16/'
    restored_path_folder_path='./paths/fundus/0'

    args.max_iter=1000000
    args.learning_rate=0.0001
    args.check_point=100
  
    input_pipeline()
    train(max_iter=args.max_iter ,learning_rate = args.learning_rate , check_point=args.check_point)