import numpy as np
import tensorflow as tf


data = np.load("train_data.npy")
filenames = tf.constant(data['dir'])
labels = tf.constant(data['price'])

# step 2: create a dataset returning slices of `filenames`
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

# step 3: parse every image in the dataset using `map`
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    return image, label

dataset = dataset.map(_parse_function)
dataset = dataset.batch(2)

# step 4: create iterator and final input tensor
iterator = dataset.make_one_shot_iterator()
images, labels = iterator.get_next()