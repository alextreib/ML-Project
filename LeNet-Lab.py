from tensorflow.examples.tutorials.mnist import input_data
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten

# Input data
mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, y_train           = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test             = mnist.test.images, mnist.test.labels

assert(len(X_train) == len(y_train))
assert(len(X_validation) == len(y_validation))
assert(len(X_test) == len(y_test))

# Pad images with 0s
X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

X_train, y_train = shuffle(X_train, y_train)

# Fixed parameters
EPOCHS = 10
BATCH_SIZE = 128

# CNN parameters
rate = 0.001

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    #Weights
    weights = {
        'f1': tf.Variable(tf.truncated_normal([5,5, 1, 6])),
        'f2': tf.Variable(tf.truncated_normal([5,5, 6, 16])),
        'f3': tf.Variable(tf.truncated_normal([5*5*16,400]))}

    biases = {
        'b1': tf.Variable(tf.truncated_normal([6])),
        'b2': tf.Variable(tf.truncated_normal([16])),
        'b3': tf.Variable(tf.truncated_normal([400]))}
    
    conv_layer = tf.nn.conv2d(x, weights['f1'], strides=[1,1,1,1], padding='VALID')
    conv_layer = tf.nn.bias_add(conv_layer, biases['b1'])

    conv_layer = tf.nn.relu(conv_layer)
    
    conv_layer = tf.nn.max_pool(
        conv_layer,
        ksize=[1,2,2,1],
        strides=[1,2,2,1],
        padding='VALID')
    
    conv_layer = tf.nn.conv2d(conv_layer, weights['f2'], strides=[1,1,1,1], padding='VALID')
    conv_layer = tf.nn.bias_add(conv_layer, biases['b2'])

    conv_layer = tf.nn.relu(conv_layer)

    conv_layer = tf.nn.max_pool(
        conv_layer,
        ksize=[1,2,2,1],
        strides=[1,2,2,1],
        padding='VALID')

    conv_layer=tf.contrib.layers.flatten(conv_layer)
    
    conv_layer=tf.contrib.layers.fully_connected(conv_layer,120)
   
    conv_layer=tf.contrib.layers.fully_connected(conv_layer,84)
    
    conv_layer=tf.contrib.layers.fully_connected(conv_layer,10)
    return conv_layer

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

def evaluate(X_data, y_data):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))