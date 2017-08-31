import tensorflow as tf
import math

import numpy as np
import sys
import os

class CNN_TFClassifier():

	def __init__(self, input_shape, num_outputs, model_dir='cnn_model', first_conv=[3, 3, 1, 32], second_conv=[3, 3, 32, 64],\
		num_epochs=50, batch_size=100, learning_rate=1e-4, dropout_prob=0.5, patience=5, output_activation='sigmoid',
		conv_strides=[1, 2, 2, 1]):
		self.input_shape = input_shape
		self.first_conv = first_conv
		self.second_conv = second_conv
		self.num_epochs = num_epochs
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.dropout_prob = dropout_prob
		self.num_outputs = num_outputs
		self.model_dir = model_dir
		self.patience = patience
		self.output_activation = output_activation
		self.conv_strides = conv_strides
		self.graph_path = os.path.join(model_dir, 'graph')
		self.training_path = os.path.join(model_dir, 'training')
		self.validation_path = os.path.join(model_dir, 'validation')
		self.model_path = os.path.join(self.graph_path, 'model.ckpt')


	def prepare_graph(self):

		with tf.name_scope("data"):
			x_ = tf.placeholder(tf.float32, shape=[None, self.input_shape[0], 
				self.input_shape[1], self.input_shape[2]], name="input")
			y_ = tf.placeholder(tf.float32, shape=[None, self.num_outputs], name="labels")

		with tf.name_scope("conv1"):
			W_conv1 = self.weight_variable(self.first_conv, name="weight1")
			b_conv1 = self.bias_variable([self.first_conv[3]], name="bias1")
			h_conv1 = tf.nn.relu(self.conv2d(x_, W_conv1) + b_conv1)
		
		with tf.name_scope("pool1"):
			h_pool1 = self.max_pool_2x2(h_conv1)

		with tf.name_scope("conv2"):
			W_conv2 = self.weight_variable(self.second_conv, name="weight2")
			b_conv2 = self.bias_variable([self.second_conv[3]], name="bias2")
			h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
			
		with tf.name_scope("pool2"):
			h_pool2 = self.max_pool_2x2(h_conv2)
		
		new_width = int(math.ceil(self.input_shape[1] / (4. * self.conv_strides[2] * self.conv_strides[2])))
		new_height = int(math.ceil(self.input_shape[0] / (4. * self.conv_strides[1] * self.conv_strides[1]) ))
		num_fully_connected = 1024

		with tf.name_scope("fc1"):
			W_fc1 = self.weight_variable([new_width * new_height * self.second_conv[3], num_fully_connected], name="weight3")
			b_fc1 = self.bias_variable([num_fully_connected], name="bias3")
			h_pool2_flat = tf.reshape(h_pool2, [-1, new_width * new_height * 64])
			h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

		with tf.name_scope("dropout1"):
			keep_prob = tf.placeholder(tf.float32)
			h_fc1_drop = tf.nn.dropout(h_fc1 , keep_prob)

		with tf.name_scope("fc2"):
			W_fc2 = self.weight_variable([num_fully_connected, self.num_outputs], name="weight4")
			b_fc2 = self.bias_variable([self.num_outputs], name="bias4")
			logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

		with tf.name_scope("loss"):
			if self.output_activation == 'sigmoid':
				cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=logits))
				y_conv_prob = tf.sigmoid(logits)
			else:
				cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))
				y_conv_prob = tf.nn.softmax(logits)

		with tf.name_scope("optimizer"):
			global_step = tf.Variable(0, name='global_step', trainable=False)
			train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy, global_step=global_step)

		net = {
			'input': x_,
			'labels': y_,
			'keep_prob': keep_prob,
			'loss': cross_entropy,
			'probabilities': y_conv_prob,
			'train_step': train_step,
			'global_step': global_step
		}
		return net


	def fit(self, x, y, x_val, y_val):
		if not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)

		g = tf.Graph()
		with g.as_default():
			net = self.prepare_graph()
			x_ = net['input']
			y_ = net['labels']
			keep_prob = net['keep_prob']
			cross_entropy = net['loss']
			y_conv_prob = net['probabilities']
			global_step = net['global_step']
			train_step = net['train_step']

			session = tf.InteractiveSession()
			indices = list(range(x.shape[0]))
			num_mini_batchs = math.floor(x.shape[0] / self.batch_size)
			max_accuracy = -1.
			session.run(tf.global_variables_initializer())

			updating_counter = 0
			accuracy_value = tf.placeholder(tf.float32, shape=())
			epoch_cost = tf.placeholder(tf.float32, shape=())
			acc_summary = tf.summary.scalar('accuracy' , accuracy_value)
			cost_summary = tf.summary.scalar('cost', epoch_cost)	
			graph_writer = tf.summary.FileWriter(self.graph_path, g)
			train_writer = tf.summary.FileWriter(self.training_path)
			val_writer = tf.summary.FileWriter(self.validation_path)
			train_merged = tf.summary.merge([cost_summary, acc_summary])
			saver = tf.train.Saver(max_to_keep=1)

			for epoch in range(1, self.num_epochs + 1):
				indices = np.random.permutation(indices)
				temp_cost = []
				for i in range(num_mini_batchs):
					first = i * self.batch_size
					last = (i + 1) * self.batch_size
					selctedIndices = indices[first : last]
					_, batch_cost = session.run([train_step, cross_entropy], feed_dict={x_: x[selctedIndices, :],\
					y_: y[selctedIndices], keep_prob: self.dropout_prob})
					temp_cost.append(batch_cost) 
				print('epoch %d finished' % epoch)
				epoch_train_accuracy = self.evaluate(x, y, x_, y_, keep_prob, y_conv_prob)
				epoch_val_accuracy = self.evaluate(x_val, y_val, x_, y_, keep_prob, y_conv_prob)
				train_summary = session.run(train_merged, 
					feed_dict={epoch_cost: np.mean(temp_cost), accuracy_value: epoch_train_accuracy})
				val_summary = session.run(acc_summary, 
					feed_dict={accuracy_value: epoch_val_accuracy})
				train_writer.add_summary(train_summary, epoch)
				train_writer.flush()				
				val_writer.add_summary(val_summary, epoch)
				val_writer.flush()

				if epoch_val_accuracy > max_accuracy:
					saver.save(session, self.model_path, global_step=global_step)
					max_accuracy = epoch_val_accuracy
					updating_counter = 0
				else:
					updating_counter += 1
				if updating_counter >= self.patience:
					break
		return max_accuracy

	def is_fitted(self):
		return tf.train.checkpoint_exists(self.graph_path)

	def predict(self, x):
		if not self.is_fitted():
			print('Model is still not fitted.')
			return None
		g = tf.Graph()
		with g.as_default():
			net = self.prepare_graph()
			x_ = net['input']
			keep_prob = net['keep_prob']
			y_conv_prob = net['probabilities']
			session = tf.InteractiveSession()
			saver = tf.train.Saver()
			saver.restore(session, tf.train.latest_checkpoint(self.graph_path))
			sliceSize = 1000
			numSlices = math.ceil(float(x.shape[0]) / sliceSize)
			outputs = np.zeros((x.shape[0], self.num_outputs), dtype=np.float64)
			for i in range(numSlices):
				first = i * sliceSize
				last = min(x.shape[0], (i + 1) * sliceSize)
				outputs[first:last] = y_conv_prob.eval(feed_dict={x_: x[first:last, :], keep_prob: 1.0})
			return outputs >= 0.5


	def evaluate(self, X, y, imagesPlaceholder, labelsPlaceholder, keepProbPlaceholder, y_output):
		sliceSize = 1000
		numSlices = math.ceil(X.shape[0] / sliceSize)
		numOutputs = y.shape[1]
		outputs = np.zeros((y.shape[0], numOutputs))
		for i in range(numSlices):
			first = i * sliceSize
			last = min(X.shape[0], (i + 1) * sliceSize)
			outputs[first: last, :] = y_output.eval(feed_dict = {imagesPlaceholder: X[first:last, :], keepProbPlaceholder: 1.0})
		if self.output_activation == 'sigmoid':
			y_pred = outputs >= 0.5
			correct_prediction = y_pred == y
			return np.mean(correct_prediction, dtype=np.float64)
		else:
			correct_prediction = np.argmax(outputs, 1) == np.argmax(y, 1)
			return np.mean(correct_prediction, dtype=np.float64)

	def weight_variable(self, shape, name):
		initial = tf.truncated_normal(shape, stddev = 0.1)
		return tf.Variable(initial, name=name)

	def bias_variable(self, shape, name):
		initial = tf.constant(0.1, shape = shape)
		return tf.Variable(initial, name=name)

	def conv2d(self, x, W):
		return tf.nn.conv2d(x, W, strides = self.conv_strides, 
			padding = 'SAME')

	def max_pool_2x2(self, x):
		return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], 
			strides = [1, 2, 2, 1], padding = 'SAME')
