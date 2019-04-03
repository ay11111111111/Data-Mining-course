import numpy as np
import tensorflow as tf


x_data = np.random.randn (2000 ,3)
w_real = [0.1, 0.4, 0.9]
b_real = -0.8
noise = np.random.randn(1, 2000) * 0.1

y_data = np.matmul(w_real, x_data.T) + b_real + noise
x = tf.placeholder(tf.float32 , shape=[None ,3])
y_true = tf.placeholder(tf.float32 , shape=None)
w = tf.Variable ([[0,0,0]],  dtype=tf.float32 , name='w')
b = tf.Variable(0, dtype=tf.float32 , name='b')

y_pred = tf.matmul(w, tf.transpose(x)) + b
loss = tf.reduce_mean(tf.square(y_true - y_pred ))
learning_rate = 0.5
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer ()
with tf.Session () as sess:
	sess.run(init)
	feed_dict = {x: x_data , y_true: y_data}
	for i in  range (1):
		sess.run(train , feed_dict)
	print(sess.run(w))

