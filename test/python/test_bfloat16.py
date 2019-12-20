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


class TestMatmulBfloat16(NgraphTest):

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
