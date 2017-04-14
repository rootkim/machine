import tensorflow as tf

x_data = [1,2,3]
y_data = [1.5, 3.5, 5.5]

W = tf.Variable(tf.random_uniform([1], -3.0, 3.0))
b = tf.Variable(tf.random_uniform([1], -3.0, 3.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X + b 

cost = tf.reduce_mean(tf.square(hypothesis - Y))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(1001):
	sess.run(train, feed_dict = {X:x_data, Y:y_data})
	if step % 50 == 0:
		print(step, sess.run(cost, feed_dict = {X:x_data, Y:y_data}))
# print("Maybe W = %s, b = %s.. " % (W, b))
print("Maybe W = %s, b = %s.. " % (sess.run(W), sess.run(b)))
print(sess.run(hypothesis, feed_dict={X:5}))
print(sess.run(hypothesis, feed_dict={X:7.5}))



