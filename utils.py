import os
import sys
import numpy as np
from scipy.misc import imread, imresize
from sklearn.model_selection import train_test_split

def read_images(path, height = 100, width = 100, is_train = True):
	fn_list = os.listdir(path)
	X = []
	Y = []
	count = 0
	for fn in fn_list:
		if fn[-3:] == 'jpg':
			count += 1
			sys.stdout.write('\r{} / {}'.format(count, len(fn_list)))
			sys.stdout.flush()

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

	return train_test_split(np.array(X), trans_to_one_hot(np.array(Y)), test_size=0.2, random_state=27)

# default binary
def trans_to_one_hot(arr,num_class=2):
	one_hot_arr = np.zeros((len(arr),num_class))
	one_hot_arr[np.arange(len(arr)),arr]=1
	return one_hot_arr
