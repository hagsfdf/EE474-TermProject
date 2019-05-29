import os
import tensorflow as tf
import numpy as np
import random
from PIL import Image
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def pool_layer(input):
    with tf.name_scope("Pool"):
        pool = tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        return pool

def BN(input, is_training, decay = 0.999):
    scale = tf.Variable(tf.ones([input.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([input.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([input.get_shape()[-1]]), trainable= False)
    pop_vari = tf.Variable(tf.ones([input.get_shape()[-1]]), trainable= False)
    epsilon = 1e-12

    if is_training == True:
        batch_mean, batch_vari = tf.nn.moments(input, list(np.arrange(0, len(input.get_shape()) -1)))
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_vari = tf.assign(pop_vari, pop_vari * decay + batch_vari * (1 - decay))
        with tf.control_dependencies([train_mean, train_vari]):
            return tf.nn.batch_normalization(input, batch_mean, batch_vari, beta, scale, epsilon)

    else:
        return tf.nn.batch_normalization(input, pop_mean, pop_vari, beta, scale, epsilon)

def fc_layer( input, size_in, size_out, dropoutProb=None, name= "FC"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.0, shape=[size_out]), name="B")
        act = tf.nn.tanh(tf.matmul(input, w) + b)

        if dropoutProb is not None:
            act = tf.nn.dropout(act, dropoutProb)

        return act

class CNN_model:

    def build(self, x, n_classes, nShape, nChannels, dropout, is_training):
        x = tf.reshape(x, [-1, nShape, nShape, nChannels])

        self.conv11 = tf.layers.conv2d(x, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                      activation=tf.nn.relu, name='Conv_11')
        self.conv11 = tf.nn.relu(BN(self.conv11, is_training))
        self.conv12 = tf.layers.conv2d(self.conv11, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                      activation=tf.nn.relu, name='Conv_12')
        self.conv12 = tf.nn.relu(BN(self.conv12, is_training))
        self.pool1 = pool_layer(self.conv12)

        self.conv21 = tf.layers.conv2d(self.pool1, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                      activation=tf.nn.relu, name='Conv_21')
        self.conv21 = tf.nn.relu(BN(self.conv21, is_training))
        self.conv22 = tf.layers.conv2d(self.conv21, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                      activation=tf.nn.relu, name='Conv_22')
        self.conv22 = tf.nn.relu(BN(self.conv22, is_training))
        self.pool2 = pool_layer(self.conv22)

        self.conv31 = tf.layers.conv2d(self.pool2, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                      activation=tf.nn.relu, name='Conv_31')
        self.conv31 = tf.nn.relu(BN(self.conv31, is_training))
        self.conv32 = tf.layers.conv2d(self.conv31, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                      activation=tf.nn.relu, name='Conv_32')
        self.conv32 = tf.nn.relu(BN(self.conv32, is_training))
        self.conv33 = tf.layers.conv2d(self.conv32, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                      activation=tf.nn.relu, name='Conv_33')
        self.conv33 = tf.nn.relu(BN(self.conv33, is_training))
        self.pool3 = pool_layer(self.conv33)



        self.conv41 = tf.layers.conv2d(self.pool3, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                      activation=tf.nn.relu, name='Conv_41')
        self.conv41 = tf.nn.relu(BN(self.conv41, is_training))
        self.conv42 = tf.layers.conv2d(self.conv41, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                       activation=tf.nn.relu, name='Conv_42')
        self.conv42 = tf.nn.relu(BN(self.conv42, is_training))
        self.conv43 = tf.layers.conv2d(self.conv42, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                       activation=tf.nn.relu, name='Conv_43')
        self.conv43 = tf.nn.relu(BN(self.conv43, is_training))
        self.pool4 = pool_layer(self.conv43)

        self.conv51 = tf.layers.conv2d(self.pool4, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                      activation=tf.nn.relu, name='Conv_51')
        self.conv51 = tf.nn.relu(BN(self.conv51, is_training))
        self.conv52 = tf.layers.conv2d(self.conv51, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                       activation=tf.nn.relu, name='Conv_52')
        self.conv52 = tf.nn.relu(BN(self.conv52, is_training))
        self.conv53 = tf.layers.conv2d(self.conv52, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                       activation=tf.nn.relu, name='Conv_53')
        self.conv53 = tf.nn.relu(BN(self.conv53, is_training))
        self.pool5 = pool_layer(self.conv53)


        print(np.shape(self.conv53))
        print(self.conv53.name)
        pool_last = self.pool5

        pool_shape = pool_last.get_shape().as_list()

        flattened = tf.reshape(pool_last, [-1, pool_shape[1] * pool_shape[2] * pool_shape[3]])
        flattened_shape = flattened.get_shape().as_list()

        self.fc1 = fc_layer(flattened, flattened_shape[1], 1024, dropoutProb=dropout, name = "FC1")
        self.fc2 = fc_layer(self.fc1, 1024, 1024, dropoutProb=dropout, name = "FC1")
        self.fc3 = fc_layer(self.fc2, 1024, n_classes, dropoutProb=dropout, name= "FC3")


        self.fc3 = tf.nn.softmax(self.fc3)
        print(self.fc3.name)

        self.Pred = self.fc3

        return self.Pred
