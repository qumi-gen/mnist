#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf


# In[2]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# In[3]:


x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels


# In[4]:


print(len(x_train))


# In[5]:


print(x_train)


# In[6]:


x = tf.placeholder(tf.float32, shape=[None,28,28])
x_image=tf.reshape(x,[-1,28,28,1])

y_ = tf.placeholder(tf.float32, shape=[None,10])


# In[7]:


x_train = np.reshape(x_train,(len(x_train),28,28))
x_test = np.reshape(x_test,(len(x_test),28,28))


# In[8]:


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)


# In[9]:


def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


# In[10]:


W_conv1=weight_variable([4,4,1,32])
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)

h_pool1 = max_pool_2x2(h_conv1)


# In[11]:


W_fc1 = weight_variable([14*14*32,80])
b_fc1 = bias_variable([80])
h_pool1_flat = tf.reshape(h_pool1,[-1,14*14*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat,W_fc1)+b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)


# In[12]:


W_fc2 = weight_variable([80,10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)


# In[13]:


cross_entropy = tf.reduce_mean(-tf.reduce_sum( y_ * tf.log(tf.clip_by_value(y_conv,1e-10,1.0)),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


# In[14]:


correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


# In[15]:


batch_size = 32
epoch_num = 10

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())


# In[23]:


for epoch in range(epoch_num):
    print(' ')
    print('epoch %d' % epoch)
    
    idx = np.random.permutation(len(x_train))
    for i in range(0, len(x_train), batch_size):
        sess. run(train_step, feed_dict={x:x_train[idx[i:i+batch_size]],y_:y_train[idx[i:i+batch_size]],keep_prob:0.5})
        train_accuracy = sess.run(accuracy,feed_dict={x:x_train[idx[i:i +batch_size]],y_:y_train[idx[i:i + batch_size]],keep_prob:1.0})
        train_loss = sess.run(cross_entropy,feed_dict={x:x_train[idx[i:i +batch_size]], y_:y_train[idx[i:i + batch_size]], keep_prob:1.0})
        print("step %d, training accuracy: %g train loss: %g" % (i, train_accuracy, train_loss))
        
    test_accuracy = sess.run(accuracy,feed_dict={x: x_test, y_: y_test, keep_prob: 1.0})
    test_loss = sess.run(cross_entropy,feed_dict={x:x_test, y_:y_test, keep_prob:1.0})
    print("---epoch %d model evaluation---" %epoch)
    print("test accuracy %f" %test_accuracy)
    print("test loss %f" %test_loss)
    print("-------------------------------------")





