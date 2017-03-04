import models as md
from utils import *


train_path = '../data/train'
test_path = '../data/test1'
width = 100
height = 100

data = read_images(train_path, height = height, width = width)
X_train, Y_train, X_eval, Y_eval = data
print X_train.shape, Y_train.shape, X_eval.shape, Y_eval.shape
# X_test, Y_test = read_images(test_path, height = height, width = width, is_train = False)
# print X_test.shape, Y_test.shape

'''
model = md.get_vgg()
md.print_model(model)

# model.fit(X_train, Y_train, batch_size=32, nb_epoch=1)

'''