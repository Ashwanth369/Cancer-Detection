# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.model_selection import train_test_split
import numpy as np
import os,cv2
import random

layers = tf.keras.layers


class Net:

	def __init__(self,batch_size = 4, num_epochs = 10, num_class = 2, learning_rate = 1e-5):

		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.num_class = num_class
		self.learning_rate = learning_rate
		
		self.image = tf.placeholder(tf.float32, shape = [None,400,400,3], name = 'input_image')
		self.ground_truth = tf.placeholder(tf.float32, shape = [None,2], name = 'class')

		X, Y = self.get_data()
		self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 42)
		# self.X_train = self.X_train[:10]
		# self.X_test = self.X_test[:10]
		# self.Y_train = self.Y_train[:10]
		# self.Y_test = self.Y_test[:10]
		
		self.output = self.build_model()
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.ground_truth, logits = self.output))
		self.train_op = tf.train.AdamOptimizer(1e-6, beta1 = 0.5, beta2 = 0.99).minimize(self.loss)
		self.sess = tf.Session()
		self.saver = tf.train.Saver(max_to_keep = 5)


	def get_data(self):

		X, Y = [], []
		for file in os.listdir('gbm-grayscale'):
			X.append('gbm-grayscale/'+file)
			Y.append([1,0])

		for file in os.listdir('lgg-grayscale'):
			X.append('lgg-grayscale/'+file)
			Y.append([0,1])

		return np.array(X),np.array(Y)


	def build_model(self):

		image = tf.reshape(self.image,(-1,400,400,3))
	
		#######################
		
		conv = layers.Conv2D(80, 10, strides = 2, padding = 'same', activation = 'relu',
							 name = 'conv1')(image)											
		lrn = tf.nn.local_response_normalization(conv, name = 'lrn1')
		pool = layers.MaxPool2D(pool_size = 6, strides = 4, name = 'maxpool1')(lrn)
		
		#######################
		
		conv = layers.Conv2D(120, 5, strides = 1, padding = 'same', activation = 'relu',
							 name = 'conv2')(pool)
		lrn = tf.nn.local_response_normalization(conv, name = 'lrn2')
		pool = layers.MaxPool2D(pool_size = 3, strides = 2, name = 'maxpool2')(lrn)

		#######################

		conv = layers.Conv2D(160, 3, strides = 1, padding = 'same', activation = 'relu',
						     name = 'conv3')(pool)

		#######################

		conv = layers.Conv2D(200, 3, strides = 1, padding = 'same', activation = 'relu',
							 name = 'conv4')(conv)
		pool = layers.MaxPool2D(pool_size = 3, strides = 2, name = 'maxpool3')(conv)

		#######################

		flat = layers.Flatten()(pool)
		fc = layers.Dense(320, activation = 'relu', name = 'fc1')(flat)
		fc = layers.Dropout(0.5)(fc)
		fc = layers.Dense(320, activation = 'relu', name = 'fc2')(fc)
		fc = layers.Dropout(0.5)(fc)
		
		output = layers.Dense(2, activation = 'softmax', name = 'output')(fc)

		return output


	def load_img(self,path):

		image = cv2.cvtColor(cv2.imread(path,-1),cv2.COLOR_BGR2RGB)
		image = cv2.resize(image,(400,400),interpolation = cv2.INTER_AREA)

		return image


	def load_batch(self,mode):

		if mode == 'train':
			X_data = self.X_train
			Y_data = self.Y_train
		elif mode == 'test':
			X_data = self.X_test
			Y_data = self.Y_test

		for batch in range(len(X_data)//self.batch_size):
			x,y = [], []
			for i in range(self.batch_size):
				x.append(self.load_img(X_data[batch*self.batch_size+i]))
				y.append(Y_data[batch*self.batch_size+i])

			yield np.array(x),np.array(y)


	def train(self):
		
		print("\n************* Training *************\n")
		self.sess.run(tf.global_variables_initializer())

		for epoch in range(self.num_epochs):

			print("Epoch: "+str(epoch))
			data_loader = self.load_batch('train')

			for i in range(len(self.X_train)//self.batch_size):
				x,y = next(data_loader)
				feed_dict = {self.image:x, self.ground_truth:y}
				self.sess.run([self.train_op,self.loss], feed_dict = feed_dict)
			
			data_loader = self.load_batch('train')
			loss_train = 0
			for i in range(len(self.X_train)//self.batch_size):
				x,y = next(data_loader)
				feed_dict = {self.image:x, self.ground_truth:y}
				loss = self.sess.run(self.loss, feed_dict = feed_dict)
				loss_train += loss

			print("\tTrain Loss: "+str(loss_train))

			data_loader = self.load_batch('test')
			loss_val = 0
			for i in range(len(self.X_test)//self.batch_size):
				x,y = next(data_loader)
				feed_dict = {self.image:x, self.ground_truth:y}
				loss = self.sess.run(self.loss, feed_dict = feed_dict)
				loss_val += loss

			print("\tTest Loss: "+str(loss_val))

			if epoch%2 == 0:
				checkpoint_path = os.path.join("checkpoints",str(epoch))
				os.makedirs(checkpoint_path)
				checkpoint_path = os.path.join(checkpoint_path, "checkpoints.ckpt")
				self.saver.save(self.sess, checkpoint_path)

		self.final_path = os.path.join("checkpoints","final") 
		os.makedirs(self.final_path)
		self.saver.save(self.sess,os.path.join(self.final_path, "checkpoints.ckpt"))

	def predict(self,image):

		self.sess.run(tf.global_variables_inititalizer())
		self.saver.restore(sess,os.path.join(self.final_path, "checkpoints.ckpt"))

		y = self.sess.run(self.output, feed_dict = {self.image:image})
		return y


if __name__ == '__main__':

	net = Net()
	net.train()