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
"""nGraph TensorFlow bridge test for assigns (possibly with validate_shape=False)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import tensorflow as tf
from common import NgraphTest


class TestAssignOperations(NgraphTest):

    def test_simple_assign(self):
        v = tf.Variable(0)
        new_v = tf.assign(v, 10)

        def run_test(sess):
            sess.run(v.initializer)
            sess.run(new_v)
            return v.eval(session=sess)

        assert self.with_ngraph(run_test) == self.without_ngraph(run_test)

