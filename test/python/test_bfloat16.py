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
"""nGraph TensorFlow bridge bfloat16 matmul operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np

import tensorflow as tf
import os

from common import NgraphTest

#This test is just a sample test to test bf16 dtype
#This fails, should enable and expand once CPU backend adds bfloat16 support

np.random.seed(5)

class TestBfloat16(NgraphTest):

    def test_conv2d_cast_bfloat16(self):
        # inputs
        input_shape_nhwc = (32, 28, 28, 3)
        filter_shape_hwio = (3, 3, 3, 16)
        input_pl = tf.placeholder(tf.float32, input_shape_nhwc, name="inp")
        filter_shape_pl = tf.placeholder(tf.float32, filter_shape_hwio, name = "out")
        input_values = np.random.rand(*input_shape_nhwc)
        filter_values = np.random.rand(*filter_shape_hwio)
        
        # cast to bloat
        input_cast = tf.cast(input_pl, dtype=tf.bfloat16)
        filter_cast = tf.cast(filter_values, dtype=tf.bfloat16)
        padding = "VALID"
        strides = [1, 1, 1, 1]
        out = tf.nn.conv2d(
            input_cast,
            filter_cast,
            strides,
            padding,
            data_format='NHWC',
            dilations=None,
            name=None)

        def run_test(sess):
            return sess.run((out,),
                            feed_dict={
                                input_pl: input_values,
                                filter_shape_pl: filter_values
                            })

        out_val = self.with_ngraph(run_test)
        print(out_val)

    def test_conv2d_cast_bfloat16(self):
        # inputs
        input_shape_nhwc = (32, 28, 28, 3)
        filter_shape_hwio = (3, 3, 3, 16)
        input_pl = tf.placeholder(tf.float32, input_shape_nhwc, name="inp")
        filter_shape_pl = tf.placeholder(tf.float32, filter_shape_hwio, name = "out")
        input_values = np.random.rand(*input_shape_nhwc)
        filter_values = np.random.rand(*filter_shape_hwio)
        
        # cast to bloat
        input_cast = tf.cast(input_pl, dtype=tf.bfloat16)
        filter_cast = tf.cast(filter_values, dtype=tf.bfloat16)
        padding = "VALID"
        strides = [1, 1, 1, 1]
        out = tf.nn.conv2d(
            input_cast,
            filter_cast,
            strides,
            padding,
            data_format='NHWC',
            dilations=None,
            name=None)

        def run_test(sess):
            return sess.run((out,),
                            feed_dict={
                                input_pl: input_values,
                                filter_shape_pl: filter_values
                            })

        out_val = self.with_ngraph(run_test)
        print(out_val)
        #assert self.with_ngraph(run_test) == self.without_ngraph(run_test)
