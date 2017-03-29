import numpy as np
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense

img_width, img_height = 150, 150

top_model_weights_path = '../data/ft_fully_connected_weights.h5'
train_data_dir = '../data/train2k'
validation_data_dir = '../data/val1k'

epochs = 100
batch_size = 32
nb_train_samples = 4000
nb_validation_samples = 1984 # need to be multiple of batches


# use vgg convolutional layers to extract features
def save_vgg_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    print 'building VGG model...'
    model = applications.VGG16(include_top=False, weights='imagenet')

    print 'extracting train features...'
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    vgg_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open('../data/vgg_features_train.npy', 'w'),
            vgg_features_train)

    print 'extract validation features...'
    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    vgg_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('../data/vgg_features_validation.npy', 'w'),
            vgg_features_validation)

# train top layers of the model
def train_top_model():
    print 'train top layers...'
    train_data = np.load(open('../data/vgg_features_train.npy'))
    train_labels = np.array([0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    validation_data = np.load(open('../data/vgg_features_validation.npy'))
    validation_labels = np.array([0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)

save_vgg_features() # just need to run this once
train_top_model()
