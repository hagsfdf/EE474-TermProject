import os
import tensorflow as tf
import numpy as np
import random
from PIL import Image
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from net import CNN_model


dir_data = "./image/"

data = np.load(dir_data + "train_data.npy")
filenames = data['dir']
labels = data['price']
N = 100  # len(labels)
n_class = 2

def label_loading():
    one_hot = np.zeros([N, n_class], dtype=np.float32)

    for i in range(N):
        label = labels[i]

        if label < 60000:
            one_hot[i, :] = [1, 0]
        elif label >= 60000:
            one_hot[i, :] = [0, 1]


    print(one_hot)
    return one_hot


def img_loading():
    for i in range(N):

        filename = filenames[i]

        if not os.path.exists(filename):
            filename = filename[:12] + '.png'
            if os.path.exists(filename):
                img = plt.imread("./" + filename)
                img = np.array(img, dtype='f')[:, :, :-1]
        else:
            img = plt.imread("./" + filename)
            img = np.array(img, dtype='f')

        img = np.reshape(img, [1, 180, 180, 3])

        if i == 0:
            img_dataset = img
        else:
            img_dataset = np.vstack([img_dataset, img])


    return img_dataset




def main():
    minibatch_size = 20
    lear_rate = 0.01
    bta1 = 0.9
    bta2 = 0.999
    epsln = 0.00000001

    input = tf.placeholder(tf.float32, shape=[None, 180, 180, 3], name='x')
    model = CNN_model()

    predicted = model.build(input, n_class, 180, 3, 0.5)

    gt = tf.placeholder(tf.float32, shape=[None, n_class], name='y')

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
        for itr in range(20001):
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
