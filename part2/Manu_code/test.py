import tensorflow as tf
import numpy as np
import sys
import scipy.misc


from utils import *
from model import *


def test(image_path, resultFoulder="test_results", outputNumber=0):

    test_image = scipy.misc.imresize(scipy.misc.imread(image_path, mode='RGB').astype('float32'),(IMAGE_HEIGHT, IMAGE_WIDTH))
    test_image = np.expand_dims(test_image, 0)
    test_image = test_image/255.

    test_data = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNEL])
    
    output = neural_network_with_skip_connections(test_data)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = initialize(sess)

    test_output = sess.run(output, feed_dict={test_data: test_image})

    scipy.misc.imsave(resultFoulder+"/%s_input_noisy.jpg"%outputNumber, test_image[0])
    scipy.misc.imsave(resultFoulder+"/%s_output_denoised.jpg"%outputNumber, test_output[0])

    print("Output saved in %s/"%resultFoulder)

if __name__ == "__main__":
    image_path = sys.argv[1]
    test(image_path)
    
