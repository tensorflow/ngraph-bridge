# ==============================================================================
#  Copyright 2018 Intel Corporation
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
"""nGraph TensorFlow bridge fusedMatMul tests.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import array_ops
from common import NgraphTest
from tensorflow.python.framework import dtypes

import numpy as np


class TestFusedMatMul(NgraphTest):
    INPUT1_SIZES = [3, 4]
    INPUT2_SIZES = [4, 6]
    BIAS_SIZES = [6]

    def test_fusedmatmul_bias(self):
        inp1_values = np.random.rand(*self.INPUT1_SIZES)
        inp2_values = np.random.rand(*self.INPUT2_SIZES)
        bias_values = np.random.rand(*self.BIAS_SIZES)

        def run_test(sess):
            inp1 = array_ops.placeholder(dtypes.float32)
            inp2 = array_ops.placeholder(dtypes.float32)
            bias = array_ops.placeholder(dtypes.float32)
            return sess.run(
                nn_ops.bias_add(tf.matmul(inp1, inp2), bias), {
                    inp1: inp1_values,
                    inp2: inp2_values,
                    bias: bias_values,
                })

        assert np.allclose(
            self.without_ngraph(run_test), self.with_ngraph(run_test))

    # FusedMatMul with Bias Relu need to be added
    @pytest.mark.skip(reason="TODO. Not supported now")
    def test_fusedmatmul_bias_relu(self):
        inp1_values = np.random.rand(*self.INPUT1_SIZES)
        inp2_values = np.random.rand(*self.INPUT2_SIZES)
        bias_values = np.random.rand(*self.BIAS_SIZES)
        def run_test(sess):
            inp1 = array_ops.placeholder(dtypes.float32)
            inp2 = array_ops.placeholder(dtypes.float32)
            bias = array_ops.placeholder(dtypes.float32)
            return sess.run(
                nn_ops.relu(
                    nn_ops.bias_add(
                         tf.matmul(
                        inp1,inp2), bias),
                {
                    inp1: inp1_values,
                    inp2: inp2_values,
                    bias: bias_values,
                }))
        assert np.allclose(
            self.without_ngraph(run_test), self.with_ngraph(run_test))
