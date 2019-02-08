import numpy as np
import tensorflow as tf
import keras.backend as K

# import for Cifar10
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.optimizers import SGD

# Import necessary components to build AlexNet
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from inception_v1 import InceptionV1

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.core.protobuf import saver_pb2

import dataset


batch_size = 32
nb_epoch = 20


def save(graph_file, ckpt_file):
    sess = K.get_session()
    saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V1)
    save_path = saver.save(sess, ckpt_file)

    gd = sess.graph.as_graph_def()
    tf.train.write_graph(gd, ".", graph_file, False)


def finetune(base_model, train_flow, test_flow, tags, train_samples_per_epoch, test_samples_per_epoch):
    nb_classes = len(tags)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(train_flow, steps_per_epoch=train_samples_per_epoch//batch_size, nb_epoch=0,
        validation_data=test_flow, validation_steps=test_samples_per_epoch//batch_size)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 'cutoff' layers and unfreeze the rest. Francois used 249
    cutoff = 0 # 150
    for layer in model.layers[:cutoff]:
        layer.trainable = False
    for layer in model.layers[cutoff:]:
        layer.trainable = True

    model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(train_flow, steps_per_epoch=train_samples_per_epoch//batch_size, nb_epoch=nb_epoch,
        validation_data=test_flow, validation_steps=test_samples_per_epoch//batch_size)


def load_data():
    X, y, tags = dataset.dataset('flowers17', 299)
    print(tags)

    nb_classes = len(tags)

    sample_count = len(y)
    train_size = sample_count * 4 // 5
    X_train = X[:train_size]
    y_train = y[:train_size]
    Y_train = to_categorical(y_train, nb_classes)
    X_test  = X[train_size:]
    y_test  = y[train_size:]
    Y_test = to_categorical(y_test, nb_classes)

    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0.125,
        height_shift_range=0.125,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest')

    datagen.fit(X_train)
    train_flow = datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True)
    test_flow = datagen.flow(X_test, Y_test, batch_size=batch_size)
    train_samples_per_epoch = X_train.shape[0]
    test_samples_per_epoch = X_test.shape[0]
    return train_flow, test_flow, tags, train_samples_per_epoch, test_samples_per_epoch


# net = VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=(299, 299, 3), pooling=None)

net = InceptionV1(include_top=False, weights='imagenet', input_tensor=None, input_shape=(299, 299, 3), pooling=None)


do_finetune = True

if do_finetune:
    train_flow, test_flow, tags, train_samples_per_epoch, test_samples_per_epoch = load_data()
    finetune(net, train_flow, test_flow, tags, train_samples_per_epoch, test_samples_per_epoch)

graph_file = "model.pb"
ckpt_file = "model.ckpt"

save(graph_file, ckpt_file)

graph_def = tf.GraphDef()
with open(graph_file, "rb") as f:
    graph_def.ParseFromString(f.read())

for node in graph_def.node:
    # if "activation" in str(node):
    print(node.name)
