import models as md
from utils import *


train_path = '../data/train1k'
test_path = '../data/test1'
width = 100
height = 100

print '\nReading training data...'
data = read_images(train_path, height = height, width = width)
X_train, X_eval, Y_train,  Y_eval = data
print '\n',X_train.shape, X_eval.shape
print Y_train.shape, Y_eval.shape

# X_test, Y_test = read_images(test_path, height = height, width = width, is_train = False)
# print X_test.shape, Y_test.shape

print '\nBuilding model...'
model = md.get_vgg()
# md.print_model(model)

print '\nStarting training...'
for i in range(50):
	model.fit(X_train, Y_train, batch_size=32, nb_epoch=1)
	score = model.evaluate(X_eval, Y_eval, batch_size=32)
	print score
