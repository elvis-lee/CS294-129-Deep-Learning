import numpy as np
import tensorflow as tf
from deeplearning.data_utils import get_CIFAR10_data
from deeplearning.layers import softmax_loss

#input labels should be one-hot encoding
X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.int64)
y_onehot = tf.one_hot(y, depth=10)
#hidden_dim = tf.placeholder(tf.int32)
mode = tf.placeholder(tf.bool) # True=training, False=test

init_he = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN')
init_x = tf.contrib.layers.xavier_initializer_conv2d()
init_x2 = tf.contrib.layers.xavier_initializer()
reg_l2 = tf.contrib.layers.l2_regularizer(0.001)
conv_relu_1 = tf.layers.conv2d(
	inputs=X,
	filters=16,
	kernel_size=(5, 5),
	strides=(1,1),
	padding="same",
	activation=tf.nn.relu,
	kernel_initializer=init_he,
	kernel_regularizer=reg_l2
	)
conv_relu_2 = tf.layers.conv2d(
	inputs=conv_relu_1,
	filters=16,
	kernel_size=(5,5),
	strides=(1,1),
	padding="same",
	activation=tf.nn.relu,
	kernel_initializer=init_he,
	kernel_regularizer=reg_l2
	)
pool3 = tf.layers.max_pooling2d(inputs=conv_relu_2, pool_size=[2, 2],strides=2)
flatten4 = tf.layers.flatten(pool3)
dense5 = tf.layers.dense(
	inputs=flatten4, 
	units=256,
	kernel_initializer=init_he,
	kernel_regularizer=reg_l2
	)
batnorm6 = tf.layers.batch_normalization(
	inputs=dense5,
	axis=1,
	training=mode,
	momentum=0.9,
	epsilon=1e-5
	)
relu7 = tf.nn.relu(batnorm6)
logits = tf.layers.dense(
	inputs=relu7,
	units=10,
	kernel_initializer=init_he,
	kernel_regularizer=reg_l2
	)

#cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits, reduction=tf.losses.Reduction.MEAN)
reg_loss = tf.losses.get_regularization_loss()
total_loss = cross_entropy + reg_loss
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
	train_step = tf.train.AdamOptimizer(1e-4).minimize(total_loss)

correct_prediction = tf.equal(tf.argmax(y_onehot, 1), tf.argmax(logits, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



# Load the (preprocessed) CIFAR10 data.
data = get_CIFAR10_data()
data['X_train'] = data['X_train'].transpose((0,2,3,1))
data['X_test'] = data['X_test'].transpose((0,2,3,1))
data['X_val'] = data['X_val'].transpose((0,2,3,1))
print data['X_train'].shape 


def next_batch(data, labels, size):
	idx = np.arange(0, len(data))
	np.random.shuffle(idx)
	idx = idx[:size]
	batch_data = data[idx]
	batch_labels = labels[idx]
	return batch_data, batch_labels

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Train
for i in range(20):
	print '============Epoch: %d============' % i
	for j in range(383):
		batch_xs, batch_ys = next_batch(data['X_train'], data['y_train'], 128)
		_, tl, cel, rl, train_accuracy, bat, den = sess.run([train_step, total_loss, cross_entropy, reg_loss, accuracy, batnorm6, dense5], feed_dict={X: batch_xs, y: batch_ys, mode: True})
		if j%20==0:
			print '--Iteration: %d' % j
			#print den[0,:10]
			#print bat[0,:10]
			print 'total loss: %f' % tl
			batch_xs_train, batch_ys_train = next_batch(data['X_train'], data['y_train'], 1000)
			print 'train accuracy: %f' % sess.run(accuracy, feed_dict={X: batch_xs_train, y: batch_ys_train, mode: False})
			print 'test accuracy: %f' % sess.run(accuracy, feed_dict={X: data['X_test'], y: data['y_test'], mode: False})

