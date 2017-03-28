from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras.optimizers import SGD

# input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
def get_vgg():
	model = Sequential()
	# this applies 32 convolution filters of size 3x3 each.
	model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(32, 32, 3)))
	model.add(Activation('relu'))
	model.add(Convolution2D(32, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Convolution2D(64, 3, 3, border_mode='valid'))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	# Note: Keras does automatic shape inference.
	model.add(Dense(256))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(2)) #10
	model.add(Activation('softmax'))

	return model

def get_simple(img_width, img_height):
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# this converts our 3D feature maps to 1D feature vectors

	model.add(Flatten())# this converts our 3D feature maps to 1D feature vectors
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	return model

def print_model(m):
	print '\nsummary',m.summary(), '\n'
	# print 'config\n',m.get_config(), '\n\n'
