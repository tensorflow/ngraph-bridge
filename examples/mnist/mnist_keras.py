# ==============================================================================
#  Copyright 2019 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================

import tensorflow as tf
import argparse
import ngraph_bridge


def prepare_data():
    # get train and test splits
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32')
    x_test = x_test.reshape(10000, 784).astype('float32')
    # One hot coding
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return x_train / 255.0, y_train, x_test / 255.0, y_test


def get_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model


def run_training(FLAGS):
    x_train, y_train, x_test, y_test = prepare_data()
    model = get_model()

    model.summary()

    # To enable ngraph with keras models, import ngraph_bridge is enough
    # For grappler builds, make sure to update_config, and set keras's session
    # You can do the update_config step on a normal, non-grappler build too
    if ngraph_bridge.is_grappler_enabled():
        config = tf.ConfigProto()
        sess = tf.Session(config=ngraph_bridge.update_config(config))
        keras.backend.set_session(sess)

    model.compile(
        loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])

    model.fit(
        x_train,
        y_train,
        batch_size=128,
        epochs=2,
        verbose=1,
        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tf_keras',
        action='store_true',
        help="If set, uses tensorflow's internal version of keras")
    parser.add_argument(
        '--epochs', type=int, default=2, help="The number of training epochs")

    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.tf_keras:
        import tensorflow.keras as keras
    else:
        import keras
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.optimizers import SGD

    run_training(FLAGS)
