import tensorflow as tf

a=tf.Variable(tf.random_normal([2,3]),name='a')
b=tf.Variable(tf.random_normal([4,5]),name='b')
c=[a,b]
shape_c=tf.shape_n(c)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(shape_c))