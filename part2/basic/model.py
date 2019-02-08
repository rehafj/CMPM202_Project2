import tensorflow as tf
import numpy as np

from utils import *

def neural_network_with_skip_connections(input_data):

    hidden_layer1 = tf.layers.conv2d(input_data, filters=32, kernel_size=3, activation=tf.nn.relu)
    hidden_layer2 = tf.layers.conv2d(hidden_layer1, filters=32, kernel_size=3, activation=tf.nn.relu)
    hidden_layer3 = tf.layers.conv2d(hidden_layer2, filters=32, kernel_size=3, activation=tf.nn.relu)
    hidden_layer4 = tf.layers.conv2d(hidden_layer3, filters=32, kernel_size=3, activation=tf.nn.relu)

    hidden_layer5 = tf.layers.conv2d_transpose(hidden_layer4, filters=32, kernel_size=3, activation=tf.nn.relu)
    hidden_layer5 = hidden_layer3 + hidden_layer5
    hidden_layer6 = tf.layers.conv2d_transpose(hidden_layer5, filters=32, kernel_size=3, activation=tf.nn.relu)
    hidden_layer6 = hidden_layer2 + hidden_layer6
    hidden_layer7 = tf.layers.conv2d_transpose(hidden_layer6, filters=32, kernel_size=3, activation=tf.nn.relu)
    hidden_layer7 = hidden_layer1 + hidden_layer7

    output_layer = tf.layers.conv2d_transpose(hidden_layer7, filters=3, kernel_size=3, activation=tf.nn.sigmoid)
    print(output_layer.shape)
    
    return output_layer

