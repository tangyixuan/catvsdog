import numpy as np
from keras import applications, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense


# global paras
train_data_dir = '../data/train'
val_data_dir = '../data/val'

train_features_dir = 'train_features.npy'
val_features_dir = 'val_features.npy'
top_model_weights = 'top_model_weights.h5'

img_width, img_height = 150, 150
nb_train_samples = 10000
nb_val_samples = 2500
epochs = 100
batch_size = 64


mode = 's'
if mode == 's':
    train_features_dir = 'train_features_s.npy'
    val_features_dir = 'val_features_s.npy'
    top_model_weights = 'top_model_weights_s.h5'

    img_width, img_height = 150, 150
    nb_train_samples = 2000
    nb_val_samples = 800
    epochs = 50
    batch_size = 16


# data augmentation config
test_datagene = ImageDataGenerator(rescale=1. / 255)
train_datagene = ImageDataGenerator(rescale=1. / 255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

def get_bottom_model(name):
    if name == 'vgg':
        return applications.VGG16(include_top=False, weights='imagenet',input_shape= (img_width,img_height,3))

def get_top_model(shape):
    model = Sequential()
    model.add(Flatten(input_shape=shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

def extract_features():
    model = get_bottom_model('vgg')

    train_data = test_datagene.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None, 
        shuffle = False)    
    
    train_features = model.predict_generator(train_data, nb_train_samples // batch_size)
    np.save(open(train_features_dir, 'w'), train_features)

    val_data = test_datagene.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle = False)    

    val_features = model.predict_generator(val_data, nb_val_samples // batch_size)
    np.save(open(val_features_dir, 'w'),val_features)


def train_fc_layers():
    # data
    train_data = np.load(open(train_features_dir))
    train_labels = np.array(
        [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    val_data = np.load(open(val_features_dir))
    val_labels = np.array(
        [0] * (nb_val_samples / 2) + [1] * (nb_val_samples / 2))
    
    # model
    model = get_top_model(train_data.shape[1:])

    model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(val_data, val_labels))
    model.save_weights(top_model_weights)

def fine_tune():
    # data 
    train_generator = train_datagene.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    val_generator = test_datagene.flow_from_directory(
        val_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    # model
    base_model = get_bottom_model('vgg')
    top_model = get_top_model(base_model.output_shape[1:])
    top_model.load_weights(top_model_weights)
    model = Model(outputs= top_model(base_model.output),inputs= base_model.input)
    
    for layer in model.layers[:15]:
        layer.trainable = False

    print model.summary()
    
    model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])

    model.fit_generator(
        train_generator,
        epochs=epochs, 
        validation_data=val_generator,
        steps_per_epoch=nb_train_samples,
        validation_steps = nb_val_samples)

# extract_features()
# train_fc_layers()
fine_tune()