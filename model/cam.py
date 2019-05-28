import os
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def cam(img_path, label):
    tf.reset_default_graph()


    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.show()


    if os.path.exists(img_path):
        if img_path[14:] == '.png':
            img = plt.imread(img_path)
            img = np.array(img, dtype='f')[:,:,:-1]

        else:
            img = plt.imread(img_path)
            img = np.array(img, dtype='f')

    img = np.reshape(img, [1, 180, 180, 3])


    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./model/model.ckpt-2200.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./model/'))

        inp = sess.graph.get_tensor_by_name("x:0")
        conv_layer  = sess.graph.get_tensor_by_name('Conv_4/Relu:0')
        fc_layer = sess.graph.get_tensor_by_name('FC2/Tanh:0')

        signal = tf.multiply(fc_layer, label)
        loss = tf.reduce_sum(signal)

        grads = tf.gradients(loss, conv_layer)[0]
        norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))

        output, grads_val = sess.run([conv_layer, norm_grads], feed_dict={inp: img})

        output = output[0]
        grads_val = grads_val[0]

        weights = np.mean(grads_val, axis=(0, 1))  # [512]
        cam = (1e-5)*np.zeros(output.shape[0: 2], dtype=np.float32)  # [7,7]

        # Taking a weighted average
        for i, w in enumerate(weights):
            cam += w * output[:, :, i]

        # Passing through ReLU
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)
        # cam = cv2.resize(cam, dsize= (180, 180))

        # Converting grayscale to 3-D
        cam3 = np.expand_dims(cam, axis=2)
        cam3 = np.tile(cam3, [1, 1, 3])

        plt.imshow(cam, cmap = 'jet')
        plt.colorbar()
        plt.show()

        output = 'output.jpg'
        heatmap = np.uint8(255* cam3)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        cv2.imwrite(output, heatmap)

    return None


def main():

    cam('./image/000003.png', [1., 0., 0., 0.])

if __name__ == '__main__':
    main()


# https://github.com/Ankush96/grad-cam.tensorflow/blob/master/main.py
