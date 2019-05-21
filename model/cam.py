import os
import tensorflow as tf
import numpy as np
import random
from PIL import Image

dir_data = "./image/"

data = np.load(dir_data + "train_data.npy")
filenames = data['dir']
labels = data['price']
N = 100  # len(labels)


def label_loading():
    one_hot = np.zeros([N, 4], dtype=np.float32)

    for i in range(N):
        label = labels[i]
        if i == 2:
            print(label)
        if label < 50000:
            one_hot[i, :] = [1, 0, 0, 0]
        elif label < 100000:
            one_hot[i, :] = [0, 1, 0, 0]
        elif label < 150000:
            one_hot[i, :] = [0, 0, 1, 0]
        elif label < 200000:
            one_hot[i, :] = [0, 0, 0, 1]

    return one_hot


def img_loading():
    for i in range(N):

        filename = filenames[i]

        if not os.path.exists(filename):
            filename = filename[:12] + '.png'
            if os.path.exists(filename):
                img = Image.open("./" + filename)
                img = np.array(img, dtype='f')[:, :, :-1]
        else:
            img = Image.open("./" + filename)
            img = np.array(img, dtype='f')

        img = np.reshape(img, [1, 180, 180, 3])

        if i == 0:
            img_dataset = img
        else:
            img_dataset = np.vstack([img_dataset, img])

    return img_dataset


def pool_layer(input):
    with tf.name_scope("Pool"):
        pool = tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        return pool


def fc_layer(input, size_in, size_out, dropoutProb=None):
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

        self.conv1 = tf.layers.conv2d(x, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                      activation=tf.nn.relu, name='Conv_1')
        self.pool1 = pool_layer(self.conv1)

        self.conv2 = tf.layers.conv2d(self.pool1, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                      activation=tf.nn.relu, name='Conv_2')
        self.pool2 = pool_layer(self.conv2)

        self.conv3 = tf.layers.conv2d(self.pool2, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                      activation=tf.nn.relu, name='Conv_3')
        self.pool3 = pool_layer(self.conv3)

        self.conv4 = tf.layers.conv2d(self.pool3, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                      activation=tf.nn.relu, name='Conv_4')

        print(self.conv4.name)
        self.pool4 = pool_layer(self.conv4)

        pool_last = self.pool4

        pool_shape = pool_last.get_shape().as_list()

        flattened = tf.reshape(pool_last, [-1, pool_shape[1] * pool_shape[2] * pool_shape[3]])
        flattened_shape = flattened.get_shape().as_list()

        self.fc1 = fc_layer(flattened, flattened_shape[1], 512, dropoutProb=dropout)
        self.fc2 = fc_layer(self.fc1, 512, n_classes, dropoutProb=dropout)

        self.fc2 = tf.nn.softmax(self.fc2)

        self.Pred = self.fc2

        return self.Pred


def main():
    minibatch_size = 20
    lear_rate = 0.00001
    bta1 = 0.9
    bta2 = 0.999
    epsln = 0.00000001

    input = tf.placeholder(tf.float32, shape=[None, 180, 180, 3], name='x')
    model = CNN_model()

    predicted = model.build(input, 4, 180, 3, 0.5)

    gt = tf.placeholder(tf.float32, shape=[None, 4], name='y')

    with tf.name_scope('Loss'):
        label_loss = tf.reduce_mean(tf.square(predicted - gt))
        tf.summary.scalar('label_loss', label_loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=lear_rate, beta1=bta1, beta2=bta2, epsilon=epsln).minimize(
        label_loss)

    saver = tf.train.Saver()

    images = img_loading()
    labeling = label_loading()

    with tf.Session() as sess:

        test_index = range(N - 3, N)
        test_data = images[test_index, :, :, :]
        test_label = labeling[test_index, :]

        train_index = range(0, N)
        train_index = [x for x in train_index if x not in test_index]
        train_data = images[train_index, :, :, :]
        train_label = labeling[train_index, :]

        sess.run(tf.global_variables_initializer())
        for itr in range(201):
            batch_idx = random.sample(range(train_data.shape[0]), minibatch_size)
            train_img_batch = train_data[batch_idx, :, :, :]
            train_lbl_batch = train_label[batch_idx, :]

            _ = sess.run([optimizer], feed_dict={input: train_img_batch, gt: train_lbl_batch})

            if itr % 100 == 0:
                [train_loss] = sess.run([label_loss], feed_dict={input: train_img_batch, gt: train_lbl_batch})
                [test_loss] = sess.run([label_loss], feed_dict={input: test_data, gt: test_label})

                print('@ iteration : %i, Training loss = %.6f, Test loss = %.6f' % (itr, train_loss, test_loss))
                saver.save(sess, os.path.join('./model', "model.ckpt"), itr)

        sess.close()


if __name__ == '__main__':
    main()


