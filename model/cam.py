import os
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def replace_none_with_zero(l):
    return [0. if i==None else i for i in l]

def cam(img_path, label, brand):
    tf.reset_default_graph()


    img = mpimg.imread(img_path)
    plt.imshow(img)
    # plt.show()

    if os.path.exists(img_path):
        if img_path[19:] == '.png':
            img = plt.imread(img_path)
            img = np.array(img, dtype='f')[:,:,:-1]
            print(np.shape(img))

        else:
            img = plt.imread(img_path)
            img = np.array(img, dtype='f')

    img = np.reshape(img, [1, 180, 180, 3])
    brand = np.reshape(brand, [1, 5])

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./model/model.ckpt-6200.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./model/'))

        inp = sess.graph.get_tensor_by_name("x:0")
        brd = sess.graph.get_tensor_by_name("Reshape_1:0")

        conv_layer  = sess.graph.get_tensor_by_name('Conv_4/Relu:0')
        fc_layer = sess.graph.get_tensor_by_name('Softmax:0')


        signal = tf.multiply(fc_layer, label)
        loss = tf.reduce_sum(signal)

        grads = tf.gradients(loss, conv_layer)[0]

        norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))

        output, grads_val = sess.run([conv_layer, norm_grads], feed_dict={inp: img, brd : brand})

        output = output[0]
        grads_val = grads_val[0]



        weights = np.mean(grads_val, axis=(0, 1))  # [256]
        cam = (1e-5)*np.ones(output.shape[0: 2], dtype=np.float32)  # [23,23]

        # Taking a weighted average
        for i, w in enumerate(weights):
            cam += w * output[:, :, i]

        # Passing through ReLU
        cam = np.maximum(cam, 0)
        if np.max(cam) == 0:
            cam = cam
        else:
            cam = cam / np.max(cam)
        cam = cv2.resize(cam, dsize= (180, 180))

        plt.imshow(cam, cmap = 'jet_r')
        plt.colorbar()
        plt.show()


        image = np.uint8(img * 255.0)  # RGB -> BGR
        cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  # balck-and-white to color
        cam = np.float32(cam) + np.float32(image)  # everlay heatmap onto the image
        cam = 255 * cam / np.max(cam)
        cam = np.uint8(cam)


        print(np.shape(cam))
        plt.imshow(cam[0])
        plt.show()
        output = 'output.jpg'

        cv2.imwrite(output, cam[0])

    return None


def main():

    cam('./imageBrand/000001.png', [0., 1., 0., 0.], [0.,0.,0.,1.,0.])

if __name__ == '__main__':
    main()


# https://github.com/Ankush96/grad-cam.tensorflow/blob/master/main.py
