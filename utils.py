import os
import numpy as np
from scipy.misc import imread, imresize
from sklearn.cross_validation import train_test_split

def read_images(path, height = 100, width = 100, is_train = True):
	fn_list = os.listdir(path)
	X = []
	Y = []
	for fn in fn_list:
		if fn[-3:] == 'jpg':
			# read in image
			file_path = os.path.join(path, fn)
			image = imread(file_path)
			image = imresize(image, (height, width))
			X.append(image)
		
			# read in label
			if is_train:
				label = fn[:fn.find('.')]
				if label == 'cat':
					Y.append(1)
				if label == 'dog':
					Y.append(0)

			# for debug
			if len(X)>99:
				break

	return train_test_split(np.array(X), np.array(Y), test_size=0.2, random_state=27)