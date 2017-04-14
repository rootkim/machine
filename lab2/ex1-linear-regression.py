import tensorflow as tf

x_data = [1,2,3]
y_data = [1.5, 3.5, 5.5]

W = tf.Variable(tf.random_uniform([1], -3.0, 3.0))
b = tf.Variable(tf.random_uniform([1], -3.0, 3.0))

hypothesis = W * x_data + b 

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(6001):
	sess.run(train)
	if step % 60 == 0:
		print(step, sess.run(W), sess.run(b))



