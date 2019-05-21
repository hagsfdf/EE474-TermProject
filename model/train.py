

import os
import tensorflow as tf
import numpy as np
import random
from PIL import Image
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def cam(img_path, label):
    tf.reset_default_graph()


    img = mpimg.imread(img_path)
    plt.imshow(img)
    # plt.show()A


    if os.path.exists(img_path):
        if img_path[14:] == '.png':
            img = Image.open(img_path)
            img = np.array(img, dtype='f')[:, :, :-1]

        else:
            img = Image.open(img_path)
            img = np.array(img, dtype='f')
    img = np.reshape(img, [1, 180, 180, 3])

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./model/model.ckpt-200.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./model/'))

        input = sess.graph.get_tensor_by_name("x:0")
        last_conv_layer = sess.graph.get_tensor_by_name('Conv_4/Relu:0')

        grad_cam = sess.run([last_conv_layer], feed_dict={input: img})

        print(np.shape(grad_cam))


    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)

    argmax = np.argmax(preds[0])  # Q1. argmax of model prediction result = 284
    output = model.output[:, argmax]

    last_conv_layer = model.get_layer(
        'block5_conv3')  # Q2. what is block number & conv. layer number for last conv. layer?
    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2)) # global average pooling
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for i in range(50):  # Q3. How many iterations?
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    hif = .4
    superimposed_img = heatmap * hif + img* 0.6  # Q4. superimpose heatmap and image
    output = 'output.jpeg'
    cv2.imwrite(output, superimposed_img)
    img = mpimg.imread(output)
    plt.imshow(img)
    plt.show()

    return None


def main():

    cam('./image/000003.png', [1, 0, 0, 0])

if __name__ == '__main__':
    main()


