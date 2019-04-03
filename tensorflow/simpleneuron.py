import  tensorflow  as tf
import  numpy  as np
import  matplotlib.pyplot  as plt
b = tf.Variable (-1.0)
w = tf.Variable (1.0)
x = tf.placeholder(tf.float32)
y = tf.nn.sigmoid(tf.add(tf.multiply(w,x),b))
print(b)
print(w)
print(x)


init = tf.global_variables_initializer()

o=np.array ([])

with tf.Session () as sess:
	sess.run(init)
	for i in np.arange (-10.0,10.0, step =0.1):
		o = np.append(o, sess.run(y, {x: i}))
	plt.plot(o)
	plt.show()

