import models as md
from utils import *

# config
img_width, img_height = 150, 150
num_epochs = 50
batch_size = 16
num_train_example = 5000

train_path = '../data/train'
test_path = '../data/eval'

print '\nReading training data...'
data = read_images(train_path, num_train_example, height = img_height, width = img_width)
X_train, X_eval, Y_train,  Y_eval = data
print '\n', X_train.shape, X_eval.shape
print Y_train.shape, Y_eval.shape

# X_test, Y_test = read_images(test_path, height = height, width = width, is_train = False)
# print X_test.shape, Y_test.shape

print '\nBuilding model...'
# model = md.get_vgg()
# model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True), metrics=['accuracy'])
model = md.get_simple(img_width,img_height)
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# md.print_model(model)

print '\nStarting training...'
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=num_epochs)
score = model.evaluate(X_eval, Y_eval, batch_size=batch_size)
print score
