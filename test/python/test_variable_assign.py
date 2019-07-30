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
"""nGraph TensorFlow bridge variables+assign test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
import numpy as np
from common import NgraphTest
import ngraph_bridge, os


class TestVarAssignOperations(NgraphTest):

    def test_simple_assign(self):
        v = tf.Variable(0)
        new_v = tf.assign(v, 10)

        def run_test(sess):
            sess.run(v.initializer)
            sess.run(new_v)
            return v.eval(session=sess)

        assert self.with_ngraph(run_test) == self.without_ngraph(run_test)


# Normal path: TF run the whole thing and we do not see any issues
# Variable and Optimizers path: Since we can see the entire graph,
# the Assign with validate_shape = False is not accepted and TF
# processes it. It is slow but functionally correct.
# Grappler path: Similar to normal path.
# Variables and Optimizers + Grappler: In this case, we cannot see the
# entire graph, hence the Var + Assign (with validate_shape=False) combo
# is not rejected. We end up accepting the var/assign in the initialize/first
# assign, but fail in the second shape shifting assign. So functionally
# incorrect.

    @pytest.mark.skipif(
        ngraph_bridge.is_grappler_enabled() and
        ngraph_bridge.are_variables_enabled(),
        reason=
        'If both grappler and variable/optimizers are enabled, this test will fail'
    )
    def test_shape_change_assign(self):
        v = tf.Variable(0)
        new_v_0 = tf.assign(v, 10)
        new_v_1 = tf.assign(v, [10, 20], validate_shape=False)

        def run_test(sess):
            sess.run(v.initializer)
            sess.run(new_v_0)
            sess.run(new_v_1)
            return v.eval(session=sess)

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()

    @pytest.mark.parametrize(("reset",), (
        (True,),
        (False,),
    ))
    def test_varassign_axpy(self, reset):
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
                        a: 1.5 * np.ones((dim1, dim2)),
                        b: np.ones((dim1, dim2))
                    })
            if reset:
                sess.run(tf.global_variables_initializer())
            return x.eval(sess)

        assert np.isclose(
            self.with_ngraph(run_test),
            (113.33008, 0)[reset] * np.ones([dim1, dim2])).all()

    def test_varassign_with_get_variable(self):
        dim1 = 3
        dim2 = 4
        a = tf.placeholder(tf.float32, shape=(dim1, dim2), name='a')
        with tf.variable_scope("foo"):  #create the first time
            x = tf.get_variable(
                "x",
                initializer=np.zeros([dim1, dim2], dtype=np.float32),
                dtype=tf.float32)
        b = tf.placeholder(tf.float32, shape=(dim1, dim2), name='y')
        c = a * x
        axpy = c + b
        train_step_0 = x.assign(axpy)
        with tf.control_dependencies([train_step_0]):
            train_op_0 = tf.no_op('train_op')

        with tf.variable_scope("foo", reuse=True):  #reuse the second time
            x_again = tf.get_variable("x")
        p = tf.placeholder(tf.float32, shape=(dim1, dim2), name='p')
        d = p * x_again
        train_step_1 = x_again.assign(d)
        with tf.control_dependencies([train_step_1]):
            train_op_1 = tf.no_op('train_op')

        def run_test(sess):
            sess.run(tf.global_variables_initializer())
            for i in range(10):
                _ = sess.run(
                    train_op_0,
                    feed_dict={
                        a: 1.5 * np.ones((dim1, dim2)),
                        b: np.ones((dim1, dim2))
                    })
            out0 = x.eval(sess)
            out1 = x_again.eval(sess)
            for i in range(2):
                _ = sess.run(
                    train_op_1, feed_dict={
                        p: 1.5 * np.ones((dim1, dim2)),
                    })
            out2 = x.eval(sess)
            out3 = x_again.eval(sess)
            return [out0, out1, out2, out3]

        arr1 = 113.33008 * np.ones([dim1, dim2])
        arr2 = 254.99268 * np.ones([dim1, dim2])
        assert np.isclose(self.with_ngraph(run_test),
                          [arr1, arr1, arr2, arr2]).all()
