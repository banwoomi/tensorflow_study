
'''
=======================================================
1. XOR backpropagation

a. sigmoid
b. ReLU
c. Leaky ReLU
d. tanh
e. Maxout
f. ELU


[ X ]--[ W1 * X + b1 ]--sigmoid--[ K : W2 * K + b2 ]--sigmoid--[ Y ]


=======================================================
'''


import numpy as np
import tensorflow as tf


# -------------------------------------------------------
#    Training Data Set
# -------------------------------------------------------
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = tf.placeholder(tf.float32, [None, 2], name='x-input')
Y = tf.placeholder(tf.float32, [None, 1], name='y-input')

#activationFunctionNumber = 1
#activationLogName = 'activation_sigmoid'

activationFunctionNumber = 2
activationLogName = 'activation_ReLU'


# -------------------------------------------------------
#   Activation Functions
# -------------------------------------------------------
def functionSelect(num, x, w, b):

    print "activationFunctionNumber ===== ", num

    if num == 1:
        print "Start Sigmoid"
        cal_value = cal_sigmoid(x, w, b)
        return cal_value

    elif num == 2:
        print "Start ReLU"
        cal_value = cal_relu(x, w, b)
        return cal_value

    else:
        cal_value = cal_sigmoid(x, w, b)
        return cal_value


# SIGMOID
def cal_sigmoid(x, w, b):
    y = tf.sigmoid(tf.matmul(x, w) + b)
    return y


# ReLU
def cal_relu(x, w, b):
    y = tf.nn.relu(tf.matmul(x, w) + b)
    return y


# -------------------------------------------------------
#   Activation Functions
# -------------------------------------------------------
def histogram(w, b, layer, num):
    tf.summary.histogram("weight" + num, w)
    tf.summary.histogram("bias" + num, b)
    tf.summary.histogram("layer" + num, layer)


with tf.name_scope("layer1") as scope:
    W1 = tf.Variable(tf.random_uniform([2, 2], -1.0, 1.0), name='weight1')
    b1 = tf.Variable(tf.zeros([2]), name='bias1')
    layer1 = functionSelect(activationFunctionNumber, X, W1, b1)
    histogram(W1, b1, layer1, '1')


with tf.name_scope("layer2") as scope:
    W2 = tf.Variable(tf.random_uniform([2, 2], -1.0, 1.0), name='weight2')
    b2 = tf.Variable(tf.zeros([2]), name='bias2')
    layer2 = functionSelect(activationFunctionNumber, layer1, W2, b2)
    histogram(W2, b2, layer2, '2')


with tf.name_scope("layer3") as scope:
    W3 = tf.Variable(tf.random_uniform([2, 2], -1.0, 1.0), name='weight3')
    b3 = tf.Variable(tf.zeros([2]), name='bias3')
    layer3 = functionSelect(activationFunctionNumber, layer2, W3, b3)
    histogram(W3, b3, layer3, '3')

with tf.name_scope("layer4") as scope:
    W4 = tf.Variable(tf.random_uniform([2, 2], -1.0, 1.0), name='weight4')
    b4 = tf.Variable(tf.zeros([2]), name='bias4')
    layer4 = functionSelect(activationFunctionNumber, layer3, W4, b4)
    histogram(W4, b4, layer4, '4')

with tf.name_scope("layer5") as scope:
    W5 = tf.Variable(tf.random_uniform([2, 2], -1.0, 1.0), name='weight5')
    b5 = tf.Variable(tf.zeros([2]), name='bias5')
    layer5 = functionSelect(activationFunctionNumber, layer4, W5, b5)
    histogram(W5, b5, layer5, '5')

with tf.name_scope("layer6") as scope:
    W6 = tf.Variable(tf.random_uniform([2, 2], -1.0, 1.0), name='weight6')
    b6 = tf.Variable(tf.zeros([2]), name='bias6')
    layer6 = functionSelect(activationFunctionNumber, layer5, W6, b6)
    histogram(W6, b6, layer6, '6')

with tf.name_scope("layer7") as scope:
    W7 = tf.Variable(tf.random_uniform([2, 2], -1.0, 1.0), name='weight7')
    b7 = tf.Variable(tf.zeros([2]), name='bias7')
    layer7 = functionSelect(activationFunctionNumber, layer6, W7, b7)
    histogram(W7, b7, layer7, '7')

with tf.name_scope("layer8") as scope:
    W8 = tf.Variable(tf.random_uniform([2, 2], -1.0, 1.0), name='weight8')
    b8 = tf.Variable(tf.zeros([2]), name='bias8')
    layer8 = functionSelect(activationFunctionNumber, layer7, W8, b8)
    histogram(W8, b8, layer8, '8')

with tf.name_scope("layer9") as scope:
    W9 = tf.Variable(tf.random_uniform([2, 2], -1.0, 1.0), name='weight9')
    b9 = tf.Variable(tf.zeros([2]), name='bias9')
    layer9 = functionSelect(activationFunctionNumber, layer8, W9, b9)
    histogram(W9, b9, layer9, '9')

with tf.name_scope("layer10") as scope:
    W10 = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0), name='weight10')
    b10 = tf.Variable(tf.zeros([1]), name='bias10')
    hypothesis = tf.sigmoid(tf.matmul(layer9, W10) + b10)
    histogram(W10, b10, hypothesis, '10')


with tf.name_scope("cost") as scope:
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
    cost_sum = tf.summary.scalar("cost", cost)


with tf.name_scope("train") as scope:
    train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)


predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
accuracy_sum = tf.summary.scalar("accuracy", accuracy)

with tf.Session() as sess:
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("../logs/xor_log/" + activationLogName)
    writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        summary, _ = sess.run([merged_summary, train], feed_dict={X: x_data, Y: y_data})
        writer.add_summary(summary, global_step=step)

        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data})
                  , sess.run([W1, W2, W3, W4, W5, W6, W7, W8, W9, W10]))

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print "Hypothesis: ", h, " Correct: ", c, "  Accuracy: ", a






