'''
---------------------------
ML Lab 10 - NN, ReLu, Xavier, Dropout, Adam
---------------------------
'''


'''
############################################
  1. MNIST Data
############################################

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Extracting MNIST_data/train-images-idx3-ubyte.gz : training set. num = 55000
# Extracting MNIST_data/train-labels-idx1-ubyte.gz : training set label.
# Extracting MNIST_data/t10k-images-idx3-ubyte.gz  : test set. num=10000
# Extracting MNIST_data/t10k-labels-idx1-ubyte.gz  : test set label.


nb_classes = 10

# Setting Variable
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

# Define Hypothesis, Cost
hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

# **** Optimizer:  GradientDescentOptimizer vs AdamOptimizer

# Accuracy
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 15
batch_size = 100

# print "number of Examples==", mnist.train.num_examples # ==>55000

# Start Training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):

        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        # print "epoch==", epoch, " total_batch==", total_batch

        for i in range(total_batch):  # 550
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch

            #if epoch == 0:
            #    print "cost==", c, " avg_cost==", avg_cost

        print 'EPOCH: %04d' % (epoch + 1), ' COST = {:.9f}' .format(avg_cost)


    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={
          X: mnist.test.images, Y: mnist.test.labels}))
    # print "mnist image:", mnist.test.images
    # print "mnist label:", mnist.test.labels
'''

'''
############################################
  2. MNIST Data (NN and Initialize with Xavier)
############################################

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Import MNIST Data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 15
batch_size = 100
display_step = 1


# Input
X = tf.placeholder(tf.float32, [None, 784])  # 28 x 28 : image of shape
Y = tf.placeholder(tf.float32, [None, 10])   # 0-9 : number

# model weights
W1 = tf.get_variable("W1", shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable("W2", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable("W3", shape=[256, 10], initializer=tf.contrib.layers.xavier_initializer())

b1 = tf.Variable(tf.random_normal([256]))
b2 = tf.Variable(tf.random_normal([256]))
b3 = tf.Variable(tf.random_normal([10]))

# Construct model
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
hypothesis = tf.matmul(L2, W3) + b3

# Define Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# **** Optimizer:  GradientDescentOptimizer vs AdamOptimizer


# Start Training
with tf.Session() as sess:

    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):

        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):  # 550
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch

        print 'EPOCH: %04d' % (epoch + 1), ' COST = {:.9f}' .format(avg_cost)

    # Test Model
    # Accuracy
    is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={
          X: mnist.test.images, Y: mnist.test.labels}))

'''



'''
############################################
  3. MNIST Data (NN and Initialize with Xavier) : more deep(layer3->layer5) and wide(256->512)
############################################

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Import MNIST Data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 15
batch_size = 100
display_step = 1


# Input
X = tf.placeholder(tf.float32, [None, 784])  # 28 x 28 : image of shape
Y = tf.placeholder(tf.float32, [None, 10])   # 0-9 : number

# model weights
W1 = tf.get_variable("W1", shape=[784, 512], initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable("W2", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable("W3", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
W4 = tf.get_variable("W4", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
W5 = tf.get_variable("W5", shape=[512, 10], initializer=tf.contrib.layers.xavier_initializer())

b1 = tf.Variable(tf.random_normal([512]))
b2 = tf.Variable(tf.random_normal([512]))
b3 = tf.Variable(tf.random_normal([512]))
b4 = tf.Variable(tf.random_normal([512]))
b5 = tf.Variable(tf.random_normal([10]))

# Construct model
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
hypothesis = tf.matmul(L4, W5) + b5

# Define Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# **** Optimizer:  GradientDescentOptimizer vs AdamOptimizer


# Start Training
with tf.Session() as sess:

    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):

        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):  # 550
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch

        print 'EPOCH: %04d' % (epoch + 1), ' COST = {:.9f}' .format(avg_cost)

    # Test Model
    # Accuracy
    is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={
          X: mnist.test.images, Y: mnist.test.labels}))

'''

'''
############################################
  4. MNIST Data (Dropout) : keep_prob: 0.5
############################################
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Import MNIST Data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 15
batch_size = 100
display_step = 1
keep_prob = tf.placeholder(tf.float32)


# Input
X = tf.placeholder(tf.float32, [None, 784])  # 28 x 28 : image of shape
Y = tf.placeholder(tf.float32, [None, 10])   # 0-9 : number

# model weights
W1 = tf.get_variable("W1", shape=[784, 512], initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable("W2", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable("W3", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
W4 = tf.get_variable("W4", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
W5 = tf.get_variable("W5", shape=[512, 10], initializer=tf.contrib.layers.xavier_initializer())

b1 = tf.Variable(tf.random_normal([512]))
b2 = tf.Variable(tf.random_normal([512]))
b3 = tf.Variable(tf.random_normal([512]))
b4 = tf.Variable(tf.random_normal([512]))
b5 = tf.Variable(tf.random_normal([10]))

# Construct model
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
hypothesis = tf.matmul(L4, W5) + b5

L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)


# Define Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# **** Optimizer:  GradientDescentOptimizer vs AdamOptimizer


# Start Training
with tf.Session() as sess:

    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):

        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):  # 550
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})
            avg_cost += c / total_batch

        print 'EPOCH: %04d' % (epoch + 1), ' COST = {:.9f}' .format(avg_cost)

    # Test Model
    # Accuracy
    is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={
          X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))


'''
############################################
  1. MNIST Data
EPOCH: 0001  COST = 1.502410240
EPOCH: 0002  COST = 0.525973589
EPOCH: 0003  COST = 0.432736704
EPOCH: 0004  COST = 0.384646428
EPOCH: 0005  COST = 0.356542665
EPOCH: 0006  COST = 0.335017520
EPOCH: 0007  COST = 0.325493728
EPOCH: 0008  COST = 0.311609380
EPOCH: 0009  COST = 0.307614288
EPOCH: 0010  COST = 0.298509303
EPOCH: 0011  COST = 0.297267001
EPOCH: 0012  COST = 0.290636185
EPOCH: 0013  COST = 0.284748050
EPOCH: 0014  COST = 0.280764486
EPOCH: 0015  COST = 0.279017345
('Accuracy: ', 0.91930002)

2. MNIST Data (NN and Initialize with Xavier) : 256
EPOCH: 0001  COST = 0.297632962
EPOCH: 0002  COST = 0.141831088
EPOCH: 0003  COST = 0.114652791
EPOCH: 0004  COST = 0.104414008
EPOCH: 0005  COST = 0.098076146
EPOCH: 0006  COST = 0.091374336
EPOCH: 0007  COST = 0.087560940
EPOCH: 0008  COST = 0.083182994
EPOCH: 0009  COST = 0.078523378
EPOCH: 0010  COST = 0.077323617
EPOCH: 0011  COST = 0.073278691
EPOCH: 0012  COST = 0.070577162
EPOCH: 0013  COST = 0.078354250
EPOCH: 0014  COST = 0.066343758
EPOCH: 0015  COST = 0.057819295
('Accuracy: ', 0.96719998)

3. MNIST Data (NN and Initialize with Xavier) : more deep(layer3->layer5) and wide(256->512)
EPOCH: 0001  COST = 0.594205068
EPOCH: 0002  COST = 0.191867400
EPOCH: 0003  COST = 0.162050258
EPOCH: 0004  COST = 0.154320981
EPOCH: 0005  COST = 0.132204037
EPOCH: 0006  COST = 0.132737760
EPOCH: 0007  COST = 0.133383421
EPOCH: 0008  COST = 0.121324443
EPOCH: 0009  COST = 0.120376196
EPOCH: 0010  COST = 0.112929258
EPOCH: 0011  COST = 0.106770817
EPOCH: 0012  COST = 0.115813755
EPOCH: 0013  COST = 0.091646599
EPOCH: 0014  COST = 0.099092963
EPOCH: 0015  COST = 0.116255206
('Accuracy: ', 0.95679998)   ==> Overfitting!!

4. MNIST Data (Dropout: Preventing overfitting)
 1) keep_prob: 0.5
EPOCH: 0001  COST = 0.500864512
EPOCH: 0002  COST = 0.186576542
EPOCH: 0003  COST = 0.172359867
EPOCH: 0004  COST = 0.141598917
EPOCH: 0005  COST = 0.134031359
EPOCH: 0006  COST = 0.150742933
EPOCH: 0007  COST = 0.112405791
EPOCH: 0008  COST = 0.110560009
EPOCH: 0009  COST = 0.117645965
EPOCH: 0010  COST = 0.112166758
EPOCH: 0011  COST = 0.115057391
EPOCH: 0012  COST = 0.098466342
EPOCH: 0013  COST = 0.099612733
EPOCH: 0014  COST = 0.100653501
EPOCH: 0015  COST = 0.091431243
('Accuracy: ', 0.96759999)

 2) keep_prob: 0.7
EPOCH: 0001  COST = 0.499725014
EPOCH: 0002  COST = 0.174490865
EPOCH: 0003  COST = 0.164388232
EPOCH: 0004  COST = 0.136852105
EPOCH: 0005  COST = 0.127595276
EPOCH: 0006  COST = 0.119552167
EPOCH: 0007  COST = 0.110716157
EPOCH: 0008  COST = 0.134227517
EPOCH: 0009  COST = 0.110341131
EPOCH: 0010  COST = 0.091627076
EPOCH: 0011  COST = 0.108611929
EPOCH: 0012  COST = 0.108955294
EPOCH: 0013  COST = 0.101021482
EPOCH: 0014  COST = 0.100737381
EPOCH: 0015  COST = 0.078319127
('Accuracy: ', 0.97170001)
############################################
'''