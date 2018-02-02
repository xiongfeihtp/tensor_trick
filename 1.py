"""
from __future__ import print_function,division

import tensorflow as tf

#graph
g=tf.Graph()
with g.as_default():
    #build graph
    a = tf.constant([[1, 2], [3, 4]],name='a')
    b = tf.constant([[3, 2], [4, 5]])
    c = tf.matmul(a, b)
    print(a.get_shape())
    print(a.dtype)
    print(a.name)
    #print(a.op)
    print(a.value_index)
    #sess
    with tf.Session() as sess:
    #operation
        print(c.eval())
"""
"""
from __future__ import  print_function,division
import numpy as np
import tensorflow as tf

v=tf.Variable(initial_value=[1,2])
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    #显式地传递session到函数里面
    v.load(value=[3,4],session=sess)
    print(v.eval(session=sess))
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Prepare train data
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10

# Define the model
#feed
X = tf.placeholder("float")
Y = tf.placeholder("float")
w = tf.Variable(0.0, name="weight")
b = tf.Variable(0.0, name="bias")
loss = tf.square(Y - X*w - b)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
for variable in tf.trainable_variables():
    print(variable)
# Create session to run
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    epoch = 1
    for i in range(10):
        for (x, y) in zip(train_X, train_Y):
            _, w_value, b_value = sess.run([train_op, w, b],feed_dict={X: x,Y: y})
        print("Epoch: {}, w: {}, b: {}".format(epoch, w_value, b_value))
        epoch += 1
#draw
plt.plot(train_X,train_Y,"+")
plt.plot(train_X,train_X.dot(w_value)+b_value)
plt.show()