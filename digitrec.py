#================================================================================================
# In this data recognition script, I will be showing how to that takes an image file             #
# containing a handwritten digit and identifies the digit using a supervised                     #
# learning algorithm and the MNIST dataset.                                                      #
# All dataset used in this script are handwritten by myself using paint on windows               #
# ============================================================================================== #
# Tutorial followed to complete and understand what to do from:                                  #
# http://dataaspirant.com/2017/05/03/handwritten-digits-recognition-tensorflow-python/,          #
# https://github.com/ianmcloughlin/jupyter-teaching-notebooks/blob/master/mnist.ipynb,           #
#  https://www.tensorflow.org/tutorials/                                                         #
#  https://www.tensorflow.org/tutorials/keras/basic_classification                               # 
#                                                                                                #
#                                                                                                #
#                                                                                                #
#=============================================================================================== #

# Using Keras to train the data from tensorflow
import tensorflow as tf
from tensorflow import keras

# Using the help of other libraries such as Numpy and Matplotlib
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# Import the handwritten dataset I wrote using paint
fashion_mnist = keras.datasets.fashion_mnist

#Handwritten_Data = keras.datasets.Handwritten_Data

# Then traing these images
(train_images, train_labels), (test_images, test_labels) = Handwritten Data.load_data()

class_names = ['Number0', 'Number1', 'Number2', 'Number3', 'Number4', 
               'Number5', 'Number6', 'Number7', 'Number8', 'Number9']

train_images.shape
len(train_labels)

train_labels
test_images.shape
len(test_labels)
