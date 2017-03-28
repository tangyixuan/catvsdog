import os
import sys
import numpy as np
from scipy.misc import imread, imresize
from sklearn.model_selection import train_test_split

def read_images(path, num_per_class, height = 100, width = 100, is_train = True):
	class_list = os.listdir(path)
	X = []
	Y = []
		
	for c in class_list:
		curr_path = os.path.join(path, c)
		fn_list = os.listdir(curr_path)
		count = 0
		if c == 'cats':
			curr_label = 0
		else:
			curr_label = 1
		for fn in fn_list:
			if fn[-3:] == 'jpg':
				count += 1
				sys.stdout.write('\r{} / {}'.format(count, num_per_class))
				sys.stdout.flush()

				# read in image
				file_path = os.path.join(curr_path, fn)
				image = imread(file_path)
				image = imresize(image, (height, width))
				X.append(image)

				# read in label
				Y.append(curr_label)
			if count> num_per_class:
				break
		
	return train_test_split(np.array(X), np.array(Y), test_size=0.2, random_state=27)
	# return train_test_split(np.array(X), trans_to_one_hot(np.array(Y)), test_size=0.2, random_state=27)

# default binary
def trans_to_one_hot(arr,num_class=2):
	one_hot_arr = np.zeros((len(arr),num_class))
	one_hot_arr[np.arange(len(arr)),arr]=1
	return one_hot_arr
