import tensorflow as tf

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a, b)
mul = tf.mul(a, b)

with tf.Session() as sess:
	print("a + b = %s" % sess.run(add, feed_dict={a: 2, b: 3}))
	print("a * b = %s" % sess.run(mul, feed_dict={a: 2, b: 3}))
