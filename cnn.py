import tensorflow as tf


def get_images(path,batch_size):
	


class CNN:

	def __init__(self,train_inputs,train_labels,test_inputs,test_labels,epochs=20,batch_size=64,name='CNN'):

		self.x = train_inputs
		self.y = train_labels

		self.xx = test_inputs
		self.yy = test_labels

		self.epochs = epochs
		self.batch_size = batch_size

		with tf.variable_scope(name):

			self.inputs_ = tf.keras.layers.InputLayer(input_shape = (400,400,3),
													  batch_size = ,  #########
													  dtype = tf.float32,
													  name='inputs')
			self.conv1 = tf.keras.layers.Conv2D(filters = 80,
										  		kernel_size = [10,10],
										  		strides = 2,
										  		padding = 'same',
										  		kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
										  		activation = 'relu',
										  		name = 'conv1')(self.inputs_)

			self.lrn1 = tf.nn.local_response_normalization(self.conv1,
														   name = 'lrn1')

			self.pool1 = tf.keras.layers.MaxPool2D(pool_size = [6,6],
												   strides = 4,
												   name = 'maxpool1')(self.lrn1)

			self.conv2 = tf.keras.layers.Conv2D(filters = 120,
										  		kernel_size = [5,5],
										  		strides = 1,
										  		padding = 'same',
										  		kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
										  		activation = 'relu',
										  		name = 'conv2')(self.pool1)

			self.lrn2 = tf.nn.local_response_normalization(self.conv2,
														   name = 'lrn2')

			self.pool2 = tf.keras.layers.MaxPool2D(pool_size = [3,3],
												   strides = 2,
												   name = 'maxpool2')(self.lrn2)
			
			self.conv3 = tf.keras.layers.Conv2D(filters = 160,
										  		kernel_size = [3,3],
										  		strides = 1,
										  		padding = 'same',
										  		kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
										  		activation = 'relu',
										  		name = 'conv3')(self.pool2)
			
			self.conv4 = tf.keras.layers.Conv2D(filters = 200,
										  		kernel_size = [3,3],
										  		strides = 1,
										  		padding = 'same',
										  		kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
										  		activation = 'relu',
										  		name = 'conv4')(self.conv3)

			self.pool3 = tf.keras.layers.MaxPool2D(pool_size = [3,3],
												   strides = 2,
												   name = 'maxpool3')(self.conv4)

			self.flat = tf.keras.layers.Flatten()(self.pool3)

			self.fc1 - tf.keras.layers.Dense(units = 320,
											 activation = 'relu',
											 name = 'fullyconnected1')(self.flat)

			self.fc1 = tf.keras.layers.Dropout(0.5)(self.fc1)

			self.fc2 = tf.keras.layers.Dense(units = 320,
											 activation = 'relu',
											 name = 'fullyconnected2')(self.fc1)

			self.fc2 = tf.keras.layers.Dropout(0.5)(self.fc2)						

			self.output = tf.keras.layers.Dense(units = 2, #########
												activation = 'softmax',
												name = 'output')(self.fc2)

	def train(self):
		
		self.model = tf.keras.Model(inputs = self.inputs_, outputs = self.output)
		self.model.compile(optimizer = 'adam',loss = tf.keras.losses.BinaryCrossentropy(),metrics = ['accuracy'])

		self.model.fit(self.x,self.y,validation_data = (self.xx,self.yy),epochs = self.epochs, batch_size = self.batch_size)