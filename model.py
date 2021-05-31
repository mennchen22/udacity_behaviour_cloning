import os.path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import Input, regularizers
from keras.applications import vgg16
from keras.engine import Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, ELU, Activation, MaxPooling2D, Conv2D, \
    BatchNormalization
from keras.models import Sequential, load_model
from keras.utils import plot_model

from data_loader import load_training_data_as_generator, generator, data_resampling, preprocessing_pipe


def test_model_on_images(model_file):
    model = load_model(model_file)
    center = cv2.imread("test_images/center.jpg")
    left = cv2.imread("test_images/left.jpg")
    right = cv2.imread("test_images/right.jpg")

    print("Expected Steering : 0.0")

    images = [left, center, right]
    label = ["Left", "Center", "Right"]
    for pos, image in enumerate(images):
        this_image = preprocessing_pipe(image=center, img_rescale=100)
        image_array = np.asarray(this_image)
        steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))
        print(label[pos] + " : " + str(steering_angle))


def vgg_net(input_shape):
    """
    Vgg net references
    - [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
    """
    model = Sequential()

    # Use batch normalization to speed up process
    model.add(BatchNormalization(input_shape=input_shape))

    # Smaller VGG Net (Blocks from VGG 16 with smaller filter depth)
    # 2 Conv, 3 Conv and 2 Conv block
    # Filter depth 8 -> 16 -> 32 (vgg16 64 --> 128 --> 256 --> 512 --> 512)
    # Dense 2048 -> 100 -> 10 --> 1 (vgg16 4096 --> 4096 --> classes)

    # Block 1
    model.add(Conv2D(8, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv 1-1'))
    model.add(Conv2D(16, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv 1-2'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 2
    # Dropout 0.2 (0.5 worse results)
    model.add(Conv2D(16, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv 2-1'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv 2-2'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv 2-3'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 3
    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv 3-1'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv 3-2'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Dense Block
    # Relu activation (ELU with no better results) --> nvidia_net_not_working()
    model.add(Flatten())
    model.add(Dense(2048))
    model.add(Activation(activation='relu'))
    model.add(Dense(100))
    model.add(Activation(activation='relu'))
    model.add(Dense(10))
    model.add(Activation(activation='relu'))
    model.add(Dense(1))

    print(model.summary())
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return model


def nvidia_net_not_working(row, col, ch):
    """
    Try to use keras vgg net with own fc layers, but results (steering from prediction) were the same for different input images
    """
    input_tensor = Input(shape=(row, col, ch))
    model_vgg16 = vgg16.VGG16(include_top=False,
                              input_tensor=input_tensor)
    x = Flatten()(model_vgg16.output)
    x = Dense(1164)(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)
    x = Dense(100)(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)
    x = Dense(50)(x)
    x = ELU()(x)
    x = Dense(10)(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)
    x = Dense(1)(x)
    # create graph of your new model
    head_model = Model(input=model_vgg16.input, output=x)
    print(head_model.summary())
    plot_model(head_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return head_model


def test_model(ch, row, col):
    """
    Test the model on three images (center/left/right) and predict the steering
    """
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Flatten())
    model.add(Dense(1))
    return model


def print_summary(history_object):
    # plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig("model_summary.png")
    plt.show()


def main_loop(data_path, img_folder="IMG"):
    image_rescale = 100
    data_path = data_path.replace("/", os.sep).replace("\\", os.sep)
    data_img_dir = os.path.join(data_path.rsplit(os.sep, 1)[0], img_folder)
    train_samples, validation_samples = load_training_data_as_generator(data_path)
    print("Number of training data {}".format(len(train_samples)))
    print("Number of validation data {}".format(len(validation_samples)))

    batch_size = 64

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size,
                                image_folder=data_img_dir, img_rescale=image_rescale)
    validation_generator = generator(validation_samples, batch_size=batch_size,
                                     image_folder=data_img_dir, img_rescale=image_rescale)

    ch = 3
    row = int(50 * (image_rescale / 100))
    col = int(320 * (image_rescale / 100))

    model = vgg_net((row, col, ch))

    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(train_generator,
                                         steps_per_epoch=int(len(train_samples) / batch_size),
                                         validation_data=validation_generator,
                                         validation_steps=int(len(validation_samples) / batch_size),
                                         epochs=10,
                                         verbose=1)
    print("Save model")
    model.save("model.h5")
    print("Done, print summary to file")
    print_summary(history_object)
    print("Finished")


if __name__ == "__main__":
    test = False
    datagen = False
    train = True

    if test:
        test_model_on_images("model_old.h5")
    if datagen:
        data_resampling("./data/track_1/driving_log.csv")
    if train:
        main_loop("./data/track_1/driving_log_to_center_only.csv")
