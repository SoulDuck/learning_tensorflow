import preprocessing
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import utils
train_imgs , train_labs , test_imgs , test_labs=preprocessing.get_cifar(type_='image')
mapping_info = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, \
                'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

mapping_info=utils.key_value_change(mapping_info)
mapping_str=utils.mapping_onehot2str(train_labs , mapping_info)


utils.plot_images(train_imgs[:100] , mapping_str)
train_imgs=np.asarray(train_imgs)
test_imgs=np.asarray(test_imgs)
train_imgs=train_imgs/255.
test_imgs=test_imgs/255.

height = 32
width = 32
color_ch=3
n_classes = 10
learning_rate=0.001
max_iter=10000
check_point=100
x_ =tf.placeholder(tf.float32, [ None , height , width , color_ch ])
y_ =tf.placeholder( tf.int32 , [ None , n_classes ])


out_ch=28
w1=tf.get_variable("w1" , [7,7,color_ch , out_ch] , initializer=tf.contrib.layers.xavier_initializer())
b1=tf.Variable(tf.constant(0.1) ,out_ch)
s1=[1,1,1,1]
p1='SAME'
layer1=tf.nn.conv2d(x_ , w1 , s1 , p1 )+b1
layer1=tf.nn.relu(layer1)


out_ch2=64
w2=tf.get_variable("w2" , [5,5,out_ch, out_ch2] , initializer=tf.contrib.layers.xavier_initializer())
b2=tf.Variable(tf.constant(0.1) ,out_ch2)
s2=[1,1,1,1]
layer2=tf.nn.conv2d(layer1, w2 , s2, padding='SAME')+b2
layer2=tf.nn.relu(layer2)
#fully connected layer = affine layer

end_conv_layer=layer2
flatten_layer=tf.contrib.layers.flatten(end_conv_layer)
length=flatten_layer.get_shape()[1]
fc_w1=tf.get_variable("fc_w1" ,[length,n_classes])
fc_b1=tf.Variable(tf.constant(0.1) , n_classes)
y_conv=tf.matmul(flatten_layer ,fc_w1 )+fc_b1


pred=tf.nn.softmax(y_conv)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv , labels=y_) , name='cost')
train=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost,name='train')
correct_pred = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1), name='correct_pred')
accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32), name='accuracy')


sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)


def next_batch(imgs , labs , batch_size):
    indices=random.sample(range(len(imgs)) , batch_size)
    batch_xs=imgs[indices]
    batch_ys=labs[indices]
    return batch_xs , batch_ys


for step in range(max_iter):
    msg='\r {}/{}'.format(step , max_iter)
    sys.stdout.write(msg)
    sys.stdout.flush()

    if step % check_point ==0:
        acc=sess.run([accuracy] , feed_dict={x_: test_imgs[:100] , y_:test_labs[:100]})
        print acc
    batch_xs , batch_ys = next_batch(train_imgs , train_labs , 60)
    sess.run([train] , feed_dict={x_:batch_xs , y_ : batch_ys })


