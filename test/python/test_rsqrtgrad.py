# ==============================================================================
#  Copyright 2018-2019 Intel Corporation
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
"""nGraph TensorFlow bridge ReluGrad operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.ops.gen_math_ops import rsqrt_grad

from common import NgraphTest
import numpy as np


class TestRsqrtGrad(NgraphTest):

    def test_rsqrtgrad_2d(self):
        y = constant_op.constant(
            self.generate_random_numbers(6, 1.0, 10.0), shape=[2, 3])
        dy = constant_op.constant(
            self.generate_random_numbers(6, 0.0, 100.0), shape=[2, 3])

        out = rsqrt_grad(y, dy)

        def run_test(sess):
            return sess.run(out)

        assert np.isclose(
            self.with_ngraph(run_test), self.without_ngraph(run_test)).all()

    def test_rsqrtgrad_1d(self):
        y = constant_op.constant(
            self.generate_random_numbers(100, 123.0, 345.0), shape=[100])
        dy = constant_op.constant(
            self.generate_random_numbers(100, 567.0, 789.0), shape=[100])

        out = rsqrt_grad(y, dy)

        def run_test(sess):
            return sess.run(out)

        assert np.isclose(
            self.with_ngraph(run_test), self.without_ngraph(run_test)).all()
