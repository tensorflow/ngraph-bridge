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
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import dtypes
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import os
import ngraph_bridge

import sys


def build_simple_model(input_array, c1, c2):
    # Convert the numpy array to TF Tensor
    input_f = tf.cast(input_array, tf.float32)

    # Define the Ops
    pl = tf.placeholder(dtype=dtypes.int32)
    pl_f = tf.cast(pl, tf.float32)
    mul = tf.compat.v1.math.multiply(input_f, c1)
    add = tf.compat.v1.math.add(mul, c2)
    add2 = add + pl_f
    output = add2
    return output, pl


def build_data_pipeline(input_array, map_function, batch_size):
    dataset = (tf.data.Dataset.from_tensor_slices(
        (tf.constant(input_array)
        )).map(map_function).batch(batch_size).prefetch(1))

    iterator = dataset.make_initializable_iterator()
    data_to_be_prefetched_and_used = iterator.get_next()

    return data_to_be_prefetched_and_used, iterator


def run_axpy_pipeline():
    input_array = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    expected_output_array = [-1, -1, 1, -1, -1, -1, -1, -1, -1]
    output_array = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    map_multiplier = 10

    map_function = lambda x: x * map_multiplier
    batch_size = 1
    pipeline, iterator = build_data_pipeline(input_array, map_function,
                                             batch_size)

    # some constants
    c1 = 5.0
    c2 = 10.0
    model, pl = build_simple_model(pipeline, c1, c2)

    with tf.Session() as sess:
        # Initialize the globals and the dataset
        sess.run(iterator.initializer)

        for i in range(1, 10):
            # Expected value is:
            # Change it to run on TF if the model gets too complex
            expected_output_array[i - 1] = (
                (input_array[i - 1] * map_multiplier) * c1) + c2 + i

            # Run one iteration
            output = sess.run(model, feed_dict={pl: i})
            output_array[i - 1] = output[0]
    return input_array, output_array, expected_output_array


def main(_):
    input_array, output_array, expected_output_array = run_axpy_pipeline()
    for i in range(1, 10):
        print("Iteration:", i, " Input: ", input_array[i - 1], " Output: ",
              output_array[i - 1], " Expected: ", expected_output_array[i - 1])
        sys.stdout.flush()


if __name__ == '__main__':
    os.environ['NGRAPH_TF_BACKEND'] = "INTERPRETER"
    #os.environ['NGRAPH_TF_USE_PREFETCH'] = "1"
    tf.app.run(main=main)
