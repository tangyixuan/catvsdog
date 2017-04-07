from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense

img_width, img_height = 128, 128

train_data_dir = '../data/train2k'
validation_data_dir = '../data/val1k'
top_model_weights_path = '../data/fc_model.h5'

nb_train_samples = 4000
nb_validation_samples = 2000
epochs = 100
batch_size = 16

# build the VGG16 network
model = applications.VGG16(weights='imagenet', include_top=False)
print('Model loaded.')
print model.summary()


# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

# start with a fully-trained classifier, including the top classifier, in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
model.add(top_model)

# set the first 25 layers to non-trainable
for layer in model.layers[:25]:
    layer.trainable = False

model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(rescale=1. / 255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_data_dir,target_size=(img_height, img_width),batch_size=batch_size,class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_data_dir,target_size=(img_height, img_width),batch_size=batch_size,class_mode='binary')

# fine-tune the model
model.fit_generator(train_generator,samples_per_epoch=nb_train_samples,epochs=epochs,validation_data=validation_generator,nb_val_samples=nb_validation_samples)