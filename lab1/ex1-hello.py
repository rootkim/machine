import tensorflow as tf

hello = tf.constant('Hello, Tensorflow...')

# start tf session
sess = tf.Session()

print(sess.run(hello))
