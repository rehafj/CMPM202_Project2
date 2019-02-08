# learning by intitioaly followin ga tutoiral at: https://medium.com/@connectwithghosh/simple-autoencoder-example-using-tensorflow-in-python-on-the-fashion-mnist-dataset-eee63b8ed9f1
# original code is credited to the authors at: https://medium.com/@connectwithghosh/simple-autoencoder-example-using-tensorflow-in-python-on-the-fashion-mnist-dataset-eee63b8ed9f1



# Importing needed libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# #
#
# # load images in cvs file seperated by ","  - ignoring first col
#
# # np.loadtxt('fashion-mnist_train.csv',\
# #                   delimiter=',', skiprows=1)[:,1:]
# #
# #
# # #loading the images
# # all_images = np.loadtxt('fashion-mnist_train.csv',\
# #                   delimiter=',', skiprows=1)[:,1:]
# # #looking at the shape of the file
# # print(all_images.shape)
