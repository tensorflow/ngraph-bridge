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
"""nGraph TensorFlow bridge varopt operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
import numpy as np
from common import NgraphTest
import ngraph_bridge, os


class TestVaroptOperations(NgraphTest):

    def test_varopt(self):
        dim1 = 3
        dim2 = 4
        a = tf.placeholder(tf.float32, shape=(dim1, dim2), name='a')
        x = tf.Variable(np.zeros([dim1, dim2]), name='x', dtype=tf.float32)
        b = tf.placeholder(tf.float32, shape=(dim1, dim2), name='y')
        c = a * x
        axpy = c + b
        train_step = x.assign(axpy)
        with tf.control_dependencies([train_step]):
            train_op = tf.no_op('train_op')

        def run_test(sess):
            sess.run(tf.global_variables_initializer())
            for i in range(10):
                _ = sess.run(
                    train_op,
                    feed_dict={
                        a: 1.5*np.ones((dim1, dim2)),
                        b: np.ones((dim1, dim2))
                    })
            return x.eval(sess)

        assert np.isclose(self.with_ngraph(run_test), 113.33008*np.ones([dim1, dim2])).all()
        


    # TODO add more tests. where sess.run runs 10 times etc


# what of reused variables?
'''
with tf.variable_scope("foo"): #create the first time
    v = tf.get_variable("v", [1])

with tf.variable_scope("foo", reuse=True): #reuse the second time
    v = tf.get_variable("v", [1])
'''
