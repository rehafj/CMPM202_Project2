import tensorflow as tf
import numpy as np
import os
import glob
import random
import scipy.misc
import cv2


N_CHANNEL = 3
LEARNING_RATE = 0.001
BATCH_SIZE = 5
N_ITERATION = 17000 #10000
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 180
IMAGE_EXT ='jpg'

TRAIN_GT_DATASET_PATH ='dataset/train/gt'
TRAIN_NOISY_DATASET_PATH ='dataset/train/noisy'
VAL_GT_DATASET_PATH ='dataset/val/gt'
VAL_NOISY_DATASET_PATH ='dataset/val/noisy'


CKPT_DIR = './Checkpoints/'
GRAPH_DIR = './Graphs/'

def changeCKPT_DIR(newFolder):
    global CKPT_DIR
    CKPT_DIR = newFolder

def initialize(sess):
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(GRAPH_DIR, sess.graph)

    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR)

    ckpt = tf.train.get_checkpoint_state(os.path.dirname(CKPT_DIR))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    return saver

def shuffle_in_unison(a, b):
     n_elem = a.shape[0]
     indices = np.random.permutation(n_elem)
     return a[indices], b[indices]


def get_random_indices(upper_limit, size):
    indices = random.sample(range(0, upper_limit), size)
    return indices

def load_data(dataset_path):
    print("loading data from " + dataset_path)

    image_array=[]

    file_list = glob.glob(dataset_path + '/*.'+IMAGE_EXT)

    temp_image_array = np.array([np.array(scipy.misc.imresize(scipy.misc.imread(file_name, mode='RGB').astype('float32'),(IMAGE_HEIGHT, IMAGE_WIDTH))) for file_name in file_list])
    temp_image_array = temp_image_array/255.

    image_array.append(temp_image_array)
    image_array = np.concatenate(image_array, axis=0)

    return image_array

def add_gaussian_noise(X_imgs):
    gaussian_noise_imgs = []
    row = X_imgs[0].shape[0]
    col = X_imgs[0].shape[1]
    channel = X_imgs[0].shape[2]

    # Gaussian distribution parameters
    mean = 0
    var = 0.1
    sigma = var ** 0.5

    for X_img in X_imgs:
        gaussian = np.random.random((row, col, channel))
        gaussian_img = cv2.addWeighted(X_img, 0.75, 0.50 * gaussian, 0.25, 0)
        gaussian_noise_imgs.append(gaussian_img)
    gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype = np.float32)
    return gaussian_noise_imgs
