import os
from skimage import data, transform
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
from skimage.color import rgb2gray
import random

# Function to load a dataset from a directory, used to import either training or testing data
def loadDataSet(directory):
    directories = [d for d in os.listdir(directory) 
                   if os.path.isdir(os.path.join(directory, d))]
    letterLabels = []
    letterImages = []
    for d in directories:
        labelDirectory = os.path.join(directory, d)
        file_names = [os.path.join(labelDirectory, f) 
                      for f in os.listdir(labelDirectory)]
        for f in file_names:
            letterImages.append(data.imread(f))
            letterLabels.append(int(d))
    return letterImages, letterLabels

# path of the project
ROOT_PATH = "/home/domenik/Documents/DAMI"

# training data directory
train_data_directory = os.path.join(ROOT_PATH, "Letters/Training")

# testing data directory
test_data_directory = os.path.join(ROOT_PATH, "Letters/Testing")

# import training data and store them in images and labels
images, labels = loadDataSet(train_data_directory)

# convert list of images to numpy array
images = np.array(images)

# convert list of labels to numpy array
labels = np.array(labels)

# remove duplicate labels
unique_labels = set(labels)

# height of the images
imageHeight = 56

# width of the images
imageWidth = 56

# amount of pixels in an image, times 4 because [r, g, b, a]
imagePixelCount = imageHeight * imageWidth * 4

# number of outputs for the neural network
outputCount = len(unique_labels)

# number of images
imageCount = int(images.size / imagePixelCount)

# number of epochs the network should be trained
epochs = 1000

# convert data to proper structure [[ color value, color value ...] [ next picture ]]
convertedImageList = [images[x].ravel() for x in range(0, imageCount)]

# convert list to numpy array
imageArray = np.array(convertedImageList)

# convert labels to proper structure, e.g. [[ 0, 0, 1], [ 1, 0, 0]]
convertedLabelList = [[ 0 for x in range(0, outputCount)] for y in range(0, labels.size)]

# convert list to numpy array
labelArray = np.array(convertedLabelList)
for x in range(0, labels.size):
    labelArray[x][labels[x]] = 1

print("ImagePixelCount: ", imagePixelCount)
print("OutputCount: ", outputCount)
print("ImageCount: ", imageCount)

# setup neural network with layer sizes corresponding to the input
x = tf.placeholder(tf.float32, [None, imagePixelCount])
W = tf.Variable(tf.zeros([imagePixelCount, outputCount]))
b = tf.Variable(tf.zeros([outputCount]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, outputCount])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# train the network with training data epochs times
for _ in range(epochs):
  sess.run(train_step, feed_dict={x: imageArray, y_: labelArray})

# calculate accuracy of the training based on training data
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: imageArray, y_: labelArray }))
