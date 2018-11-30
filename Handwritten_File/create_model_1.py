
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
# https://github.com/docketrun/Recognise-Handwritten-Digits-using-MNIST-Data                     # 
#                                                                                                #
#                                                                                                #
#                                                                                                #
#=============================================================================================== #

#import modules
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#import data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init_op = tf.initialize_all_variables()
saver = tf.train.Saver()


# Train the model and save the model to disk as a model.ckpt file
# file is stored in the same directory as this python script is started
"""
The use of 'with tf.Session() as sess:' is taken from the Tensor flow documentation
on on saving and restoring variables.
https://www.tensorflow.org/versions/master/how_tos/variables/index.html
"""
with tf.Session() as sess:
    sess.run(init_op)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        
    save_path = saver.save(sess, "./model1.ckpt")
    print ("Model saved in file: ", save_path)

