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


	def prepare_graph(self):
		imageSize = self.input_shape[0] * self.input_shape[1]

		x_ = tf.placeholder(tf.float32, shape=[None, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
		y_ = tf.placeholder(tf.float32, shape=[None, self.num_outputs])

		W_conv1 = self.weight_variable(self.first_conv)
		b_conv1 = self.bias_variable([self.first_conv[3]])

		h_conv1 = tf.nn.relu(self.conv2d(x_, W_conv1) + b_conv1)
		h_pool1 = self.max_pool_2x2(h_conv1)

		W_conv2 = self.weight_variable(self.second_conv)
		b_conv2 = self.bias_variable([self.second_conv[3]])

		h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
		h_pool2 = self.max_pool_2x2(h_conv2)
		
		new_width = int(math.ceil(self.input_shape[1] / (4. * self.conv_strides[2] * self.conv_strides[2])))
		new_height = int(math.ceil(self.input_shape[0] / (4. * self.conv_strides[1] * self.conv_strides[1]) ))

		W_fc1 = self.weight_variable([new_width * new_height * self.second_conv[3], 1024])
		b_fc1 = self.bias_variable([1024])

		h_pool2_flat = tf.reshape(h_pool2, [-1, new_width * new_height * 64])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

		keep_prob = tf.placeholder(tf.float32)
		h_fc1_drop = tf.nn.dropout(h_fc1 , keep_prob)

		W_fc2 = self.weight_variable([1024, self.num_outputs])
		b_fc2 = self.bias_variable([self.num_outputs])

		y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
		
		if self.output_activation == 'sigmoid':
			cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_conv, y_))
			y_conv_prob = tf.sigmoid(y_conv)
		else:
			cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
			y_conv_prob = tf.nn.softmax(y_conv)
		wrong_prediction = tf.not_equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
		MAE = tf.reduce_mean(tf.cast(wrong_prediction, tf.float32))
		return (x_, y_, keep_prob, cross_entropy, y_conv, y_conv_prob)


	def fit(self, x, y, x_val, y_val):
		if not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)
		g = tf.Graph()
		with g.as_default():
			x_, y_, keep_prob, cross_entropy, y_conv, y_conv_prob = self.prepare_graph()
			session = tf.InteractiveSession()
			indices = list(range(x.shape[0]))
			num_mini_batchs = math.floor(x.shape[0] / self.batch_size)
			train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)
			
			file_path = self.model_dir + '/training.txt'
			last_model_epoch = self.last_epoch(file_path)
			start_index = last_model_epoch + 1
			max_accuracy = -1.
			if last_model_epoch >= 1:
				self.restore_model(session)
				max_accuracy = self.evaluate(x_val, y_val, x_, y_, keep_prob, y_conv_prob)
			else:
				session.run(tf.initialize_all_variables())

			if not os.path.exists(file_path):
				append_to_file(file_path, 'epoch\ttrain_cost\ttrain_acc\tval_acc\n')
				
			print('Start training from epoch %d' % start_index)
			# correct_prediction = tf.equal(tf.argmax(tf.y_conv.eval(feed_dict = {x_: x_val, keep_prob: 1.0}), 1), np.argmax(y_val, 1))
			# val_accuracy = tf.reduce_mean(correct_prediction)
			updating_counter = 0
			for epoch in range(start_index, self.num_epochs + 1):
				indices = np.random.permutation(indices)
				epochCost = []
				for i in range(num_mini_batchs):
					first = i * self.batch_size
					last = (i + 1) * self.batch_size
					selctedIndices = indices[first : last]
					_, batchCost = session.run([train_step, cross_entropy], feed_dict={x_: x[selctedIndices, :],\
					y_: y[selctedIndices], keep_prob: self.dropout_prob})
					epochCost.append(batchCost) 
				print('epoch %d finished' % epoch)
				mean_cost = np.mean(epochCost)
				epoch_train_accuracy = self.evaluate(x, y, x_, y_, keep_prob, y_conv_prob)
				epoch_val_accuracy = self.evaluate(x_val, y_val, x_, y_, keep_prob, y_conv_prob)

				print("training accuracy: %f --- validation accuracy: %f " % (epoch_train_accuracy, epoch_val_accuracy))
				# tf.summary.scalar('val_accuracy', epoch_train_accuracy)
				if epoch_val_accuracy > max_accuracy:
					self.save_model(session)
					max_accuracy = epoch_val_accuracy
					updating_counter = 0
				else:
					updating_counter += 1
				append_to_file(file_path, '%d\t%f\t%f\t%f\n' % (epoch, mean_cost, epoch_train_accuracy, epoch_val_accuracy))
				if updating_counter >= self.patience:
					break
		return max_accuracy

	def is_fitted(self):
		file_path = self.model_dir + '/training.txt'
		return self.last_epoch(file_path) < 1

	def predict(self, x):
		file_path = self.model_dir + '/training.txt'
		last_model_epoch = self.last_epoch(file_path)
		if self.is_fitted():
			print('Model is still not fitted.')
			return None
		g = tf.Graph()
		with g.as_default():
			x_, y_, keep_prob, cross_entropy, y_conv, y_conv_prob = self.prepare_graph()
			session = tf.InteractiveSession()
			self.restore_model(session)

			sliceResults = []
			sliceSize = 50
			numSlices = math.ceil(float(x.shape[0]) / sliceSize)
			outputs = np.zeros((x.shape[0], self.num_outputs), dtype=np.float64)
			for i in range(numSlices):
				first = i * sliceSize
				last = min(x.shape[0], (i + 1) * sliceSize)
				outputs[first:last] = y_conv_prob.eval(feed_dict = {x_: x[first:last, :], keep_prob: 1.0})
			return outputs


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
			
	def save_model(self, session):
		if not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)
		saver = tf.train.Saver()
		saver.save(session, '%s/model.ckpt' % self.model_dir)

	def restore_model(self, session):
		saver = tf.train.Saver()
		saver.restore(session, ('%s/model.ckpt' % self.model_dir))

	def last_epoch(self, file_path):
		if os.path.exists(file_path):
			num_lines = sum(1 for line in open(file_path))
			if num_lines > 1:
				mat = np.loadtxt(file_path, delimiter='\t', skiprows = 1, ndmin=2)
				return int(mat[mat.shape[0] - 1, 0])
		return 0

	def weight_variable(self, shape):
		initial = tf.truncated_normal(shape, stddev = 0.1)
		return tf.Variable(initial)

	def bias_variable(self, shape):
		initial = tf.constant(0.1, shape = shape)
		return tf.Variable(initial)

	def conv2d(self, x, W):
		return tf.nn.conv2d(x, W, strides = self.conv_strides, 
			padding = 'SAME')

	def max_pool_2x2(self, x):
		return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], 
			strides = [1, 2, 2, 1], padding = 'SAME')

def append_to_file(file_path, text):
	with open(file_path, 'a') as file:
		file.write(text)