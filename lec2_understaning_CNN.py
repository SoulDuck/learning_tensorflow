#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import preprocessing
import os
import matplotlib

if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
import random
import sys
import utils
import pickle

def lec2_1():
    matrix =np.asarray([[0,0,0],[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5]] , dtype=np.float32)
    matrix=matrix.reshape([1,6,3,1])
    filter=np.asarray([[0,1],[1,0]] , dtype=np.float32)
    filter=filter.reshape([2,2,1,1])

    t_filter=tf.Variable(filter)
    t_matrix=tf.Variable(matrix)
    strides=[1,1,1,1]

    output=tf.nn.conv2d(t_matrix , t_filter , strides , padding='SAME')
    init = tf.global_variables_initializer()
    sess=tf.Session()
    sess.run(init)
    output_=sess.run(output)
    print 'matrix'
    print np.squeeze(matrix)
    print 'filter'
    print np.squeeze(filter)
    print 'output'
    print np.squeeze(output_)


def lec2_2():
    matrix =np.asarray([[0,0,0],[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5]] , dtype=np.float32)
    matrix=matrix.reshape([1,6,3,1])
    filter=np.asarray([[0,1],[1,0]] , dtype=np.float32)
    filter=filter.reshape([2,2,1,1])

    t_filter=tf.Variable(filter)
    t_matrix=tf.Variable(matrix)
    strides=[1,1,1,1]

    output=tf.nn.conv2d(t_matrix , t_filter , strides , padding='VALID')
    init = tf.global_variables_initializer()
    sess=tf.Session()
    sess.run(init)
    output_=sess.run(output)
    print 'matrix'
    print np.squeeze(matrix)
    print 'filter'
    print np.squeeze(filter)
    print 'output'
    print np.squeeze(output_)



def lec2_3():
    train_imgs, train_labs, test_imgs, test_labs = preprocessing.get_cifar_data()

    def next_batch(imgs, labs, batch_size):
        indices = random.sample(range(len(imgs)), batch_size)
        batch_xs = imgs[indices]
        batch_ys = labs[indices]
        return batch_xs, batch_ys
    height = 32
    width = 32
    color_ch=3
    n_classes = 10
    learning_rate=0.001
    max_iter=10000
    check_point=100
    x_ =tf.placeholder(tf.float32, [ None , height , width , color_ch ])
    y_ =tf.placeholder( tf.int32 , [ None , n_classes ])

    # Placeholder 는 차후에 입력할 값
    # Variables 는 weight 로 학습시 변하는 값
    out_ch=28

    w1=tf.get_variable("w1" , [11,11,color_ch , out_ch] , initializer=tf.contrib.layers.xavier_initializer())
    b1=tf.Variable(tf.constant(0.1) ,out_ch)
    s1=[1,1,1,1]
    p1='SAME'
    layer1=tf.nn.conv2d(x_ , w1 , s1 , p1 )+b1
    layer1=tf.nn.relu(layer1)


    out_ch2=64
    w2=tf.get_variable("w2" , [7,7,out_ch, out_ch2] , initializer=tf.contrib.layers.xavier_initializer())
    b2=tf.Variable(tf.constant(0.1) ,out_ch2)
    s2=[1,1,1,1]
    layer2=tf.nn.conv2d(layer1, w2 , s2, padding='SAME')+b2
    layer2=tf.nn.relu(layer2)


    pool_s = [1,2,2,1]
    layer3 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=pool_s, padding='SAME')

    out_ch3=128
    w3=tf.get_variable("w3" , [5,5,out_ch2, out_ch3] , initializer=tf.contrib.layers.xavier_initializer())
    b3=tf.Variable(tf.constant(0.1) ,out_ch3)
    s3=[1,1,1,1]
    layer4=tf.nn.conv2d(layer3, w3 , s3, padding='SAME')+b3
    layer4=tf.nn.relu(layer4)

    out_ch4=128
    w4=tf.get_variable("w4" , [5,5,out_ch3, out_ch4] , initializer=tf.contrib.layers.xavier_initializer())
    b4=tf.Variable(tf.constant(0.1) ,out_ch4)
    s4=[1,1,1,1]
    layer4=tf.nn.conv2d(layer4, w4 , s4, padding='SAME')+b4
    layer4=tf.nn.relu(layer4)

    layer5 = tf.nn.max_pool(layer4, ksize=[1, 2, 2, 1], strides=pool_s, padding='SAME')

    #fully connected layer = affine layer

    end_conv_layer=layer5
    flatten_layer=tf.contrib.layers.flatten(end_conv_layer)
    length=flatten_layer.get_shape()[1]
    fc_w1=tf.get_variable("fc_w1" ,[length,n_classes])
    fc_b1=tf.Variable(tf.constant(0.1) , n_classes)
    y_conv=tf.matmul(flatten_layer ,fc_w1 )+fc_b1

    pred=tf.nn.softmax(y_conv)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= pred , labels=y_) , name='cost')
    train=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost,name='train')
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1), name='correct_pred')
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32), name='accuracy')


    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()

    batch_iteration = 100
    training_epochs = 2000

    train_cost_list = []
    test_cost_list = []
    train_acc_list = []
    test_acc_list = []

    for epoch in range(training_epochs):

        train_avg_cost = 0.
        train_avg_acc = 0.
        test_avg_cost = 0.
        test_avg_acc = 0.

        for batch in range(batch_iteration):
            batch_xs, batch_ys = next_batch(train_imgs, train_labs, 60)

            sess.run(train, feed_dict={x_: batch_xs, y_: batch_ys})
            train_avg_cost += sess.run(cost, feed_dict={x_: batch_xs, y_: batch_ys})
            train_avg_acc += sess.run(accuracy, feed_dict={x_: batch_xs, y_: batch_ys})

        train_avg_cost = train_avg_cost / batch_iteration
        train_avg_acc = train_avg_acc / batch_iteration

        test_avg_cost = sess.run(cost, feed_dict={x_: test_imgs, y_: test_labs})
        test_avg_acc = sess.run(accuracy, feed_dict={x_: test_imgs, y_: test_labs})

        print "##################################"
        print ("Epoch: %03d/%03d train cost: %.4f" % (epoch, training_epochs, train_avg_cost))
        print ("Epoch: %03d/%03d train acc: %.4f" % (epoch, training_epochs, train_avg_acc))
        print ("Epoch: %03d/%03d test cost: %.4f" % (epoch, training_epochs, test_avg_cost))
        print ("Epoch: %03d/%03d test acc: %.4f" % (epoch, training_epochs, test_avg_acc))

        train_cost_list.append(train_avg_cost)
        test_cost_list.append(test_avg_cost)
        train_acc_list.append(train_avg_acc)
        test_acc_list.append(test_avg_acc)

    saver.save(sess, "model/cifar_deep_convolution.ckpt")

    with open('cost_acc/cifar_deep_convolution_train_cost', 'wb') as fp:
        pickle.dump(train_cost_list, fp)
    with open('cost_acc/cifar_deep_convolution_test_cost', 'wb') as fp:
        pickle.dump(test_cost_list, fp)
    with open('cost_acc/cifar_deep_convolution_train_acc', 'wb') as fp:
        pickle.dump(train_acc_list, fp)
    with open('cost_acc/cifar_deep_convolution_test_acc', 'wb') as fp:
        pickle.dump(test_acc_list, fp)
    # softmax : 결과값을 각 class 에 해당하는 확률값을로 리턴
    # softmax_cross_entropy_with_logits : 실제 정답인 onehot vector 와 예측값 pred 를 차이를 cross_entropy 로 계산
    # tf.train.GradientDescentOptimizer : cost 가 최소와 되도록 weight를 조정하는 함수
    # accuracy : 실제 값과 예측값의 일치률
    # 4개의 convolution neural network 와 1개의 fully connected_layer 로 구성
    # 2개의 convolution layer 를 거친 후 각 각 max pooling 적용 / max pooling 후에는 activation map 의 가로 세로 크기가 절반이 된다.
    # stride : 좌우로 몇 칸 씩 커널을 이동 할 것인지에 대한 값
    # padding : convolution 전후로 activation map 의 크기를 조정하기 위한 값 , SAME 을 입력하면 항상 convolution 전후의 크기가 같다.
    # weight : [커널의 가로크기, 커널의 세로크기, input activation map 의 채널 크기, output activiation map 의 채널 크기]
if __name__=='__main__':
    lec2_3()
