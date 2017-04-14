import tensorflow as tf

# Start tf session
sess = tf.Session()

a = tf.constant(2)
b = tf.constant(3)

c = a + b

# print out operation
print(c) 

# print out the result of operation
print(sess.run(c))
