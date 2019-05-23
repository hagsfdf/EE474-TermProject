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


def fc_layer( input, size_in, size_out, dropoutProb=None):
    with tf.name_scope("FC"):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.0, shape=[size_out]), name="B")
        act = tf.nn.tanh(tf.matmul(input, w) + b)

        if dropoutProb is not None:
            act = tf.nn.dropout(act, dropoutProb)

        return act

class CNN_model:

    def build(self, x, n_classes, nShape, nChannels, dropout):
        x = tf.reshape(x, [-1, nShape, nShape, nChannels])

        self.conv11 = tf.layers.conv2d(x, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                      activation=tf.nn.relu, name='Conv_11')

        self.conv12 = tf.layers.conv2d(self.conv11, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                      activation=tf.nn.relu, name='Conv_12')

        self.pool1 = pool_layer(self.conv12)

        self.conv21 = tf.layers.conv2d(self.pool1, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                      activation=tf.nn.relu, name='Conv_21')

        self.conv22 = tf.layers.conv2d(self.conv21, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                      activation=tf.nn.relu, name='Conv_22')

        self.pool2 = pool_layer(self.conv21)

        self.conv31 = tf.layers.conv2d(self.pool2, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                      activation=tf.nn.relu, name='Conv_31')


        self.conv32 = tf.layers.conv2d(self.conv31, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                      activation=tf.nn.relu, name='Conv_32')
        self.pool3 = pool_layer(self.conv31)


        self.conv4 = tf.layers.conv2d(self.pool3, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                      activation=tf.nn.relu, name='Conv_4')

        print(self.conv4.name)
        print(np.shape(self.conv4))
        self.pool4 = pool_layer(self.conv4)

        pool_last = self.pool4

        pool_shape = pool_last.get_shape().as_list()

        flattened = tf.reshape(pool_last, [-1, pool_shape[1] * pool_shape[2] * pool_shape[3]])
        flattened_shape = flattened.get_shape().as_list()

        self.fc1 = fc_layer(flattened, flattened_shape[1], 512, dropoutProb=dropout)
        self.fc2 = fc_layer(self.fc1, 512, n_classes, dropoutProb=dropout)

        print(self.fc2.name)
        self.fc2 = tf.nn.softmax(self.fc2)

        self.Pred = self.fc2

        return self.Pred
