import numpy as np
import tensorflow as tf

tf.set_random_seed(777)

learning_rate=0.1

x=np.array([[0,0],
   [0,1],
   [1,1],
   [1,0]],dtype=np.float32)

y=np.array([[0],
   [1],
   [0],
   [1]],dtype=np.float32)

X=tf.placeholder(tf.float32,[None,2])
Y=tf.placeholder(tf.float32,[None,1])

w=tf.Variable(tf.random_normal([2,1]),name='weight')
b=tf.Variable(tf.random_normal([1]),name='bias')

hp=tf.sigmoid(tf.matmul(X,w)+b)

cost=-tf.reduce_mean(Y*tf.log(hp)+(1-Y)*tf.log(1-hp))

train=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

predicted = tf.cast(hp>0.5,dtype=tf.float32)

accuracy=tf.reduce_mean(tf.cast(tf.equal(predicted,Y),dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for setp in range(10001):
        sess.run(train,feed_dict={X:x,Y:y})

    h,c,a=sess.run([hp,predicted,accuracy],feed_dict={X:x,Y:y})
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)