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

    @pytest.mark.skip(reason="CPU backend does not support dtype bf16")
    def test_matmul_bfloat16(self):
        a = tf.placeholder(tf.bfloat16, [2, 3], name='a')
        x = tf.placeholder(tf.bfloat16, [3, 4], name='x')
        a_inp = np.random.rand(2, 3)
        x_inp = np.random.rand(3, 4)
        out = tf.matmul(a, x)

        def run_test(sess):
            return sess.run((out,), feed_dict={a: a_inp, x: x_inp})

        assert self.with_ngraph(run_test) == self.without_ngraph(run_test)

    def test_conv2d_cast_bfloat16(self):
        # inputs
        input_shape_nhwc = (1, 8, 8, 1)
        filter_shape_hwio = (3, 3, 1, 2)
        input_pl = tf.placeholder(tf.float32, input_shape_nhwc, name="inp_pl")
        filter_shape_pl = tf.placeholder(
            tf.float32, filter_shape_hwio, name="filter_pl")
        input_values = np.arange(64).reshape(
            input_shape_nhwc)  #np.random.rand(*input_shape_nhwc)
        filter_values = np.arange(18).reshape(
            filter_shape_hwio)  # np.random.rand(*filter_shape_hwio)
        print(filter_values)
        # cast to bloat
        input_cast = tf.cast(input_pl, dtype=tf.bfloat16)
        filter_cast = tf.cast(filter_shape_pl, dtype=tf.bfloat16)
        padding = "VALID"
        strides = [1, 1, 1, 1]
        conv_op = tf.nn.conv2d(
            input_cast,
            filter_cast,
            strides,
            padding,
            data_format='NHWC',
            dilations=None,
            name=None)
        out = tf.cast(conv_op, dtype=tf.float32)

        def run_test(sess):
            return sess.run((conv_op,),
                            feed_dict={
                                input_pl: input_values,
                                filter_shape_pl: filter_values
                            })

        assert np.allclose(
            self.with_ngraph(run_test), self.without_ngraph(run_test))
