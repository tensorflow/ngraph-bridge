# ==============================================================================
#  Copyright 2018-2020 Intel Corporation
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
"""nGraph TensorFlow bridge range operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import os
import numpy as np

from common import NgraphTest


class TestRangeOperations(NgraphTest):

    def test_range_constant(self):
        start = tf.constant(2, dtype=tf.int32, shape=[])
        end = tf.constant(7, dtype=tf.int32, shape=[])
        out = tf.range(start, end)

        def run_test(sess):
            return sess.run(out)

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()
