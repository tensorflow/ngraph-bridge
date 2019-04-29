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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
import os
import numpy as np
import ngraph_bridge
import sys

from common import NgraphTest


class TestOpDisableOperations(NgraphTest):

    # Note it is possible to set an invalid op name (as long as mark_for_clustering is not called)
    @pytest.mark.parametrize(("op_list",), (('Add',), ('Add,Sub',), ('',),
                                            ('_InvalidOp',)))
    def test_disable_op_1(self, op_list):
        ngraph_bridge.set_disabled_ops(op_list)
        assert ngraph_bridge.get_disabled_ops() == op_list.encode("utf-8")
        # Running get_disabled_ops twice to see nothing has changed between 2 consecutive calls
        assert ngraph_bridge.get_disabled_ops() == op_list.encode("utf-8")

    # Test to see that exception is raised if sess.run is called with invalid op types
    @pytest.mark.parametrize(("invalid_op_list",), (('Add,_InvalidOp',),
                                                    ('NGraphEncapsulate',)))
    def test_disable_op_2(self, invalid_op_list):
        ngraph_bridge.set_disabled_ops(invalid_op_list)
        a = tf.placeholder(tf.int32, shape=(5,))
        b = tf.constant(np.ones((5,)), dtype=tf.int32)
        c = a + b

        def run_test(sess):
            return sess.run(c, feed_dict={a: np.ones((5,))})

        assert (self.without_ngraph(run_test) == np.ones(5,) * 2).all()
        try:
            # This test is expected to fail, since all the strings passed to set_disabled_ops have invalid ops in them
            res = self.with_ngraph(run_test)
        except:
            return
        assert False, 'Had expected test to raise error'

    def test_disable_3(self):
        # TODO: make the env var setting and resetting a decorator
        log_placement = os.environ.pop('NGRAPH_TF_LOG_PLACEMENT', None)
        os.environ['NGRAPH_TF_LOG_PLACEMENT'] = '1'
        a = tf.placeholder(tf.int32, shape=(5,))
        b = tf.constant(np.ones((5,)), dtype=tf.int32)
        ngraph_bridge.set_disabled_ops('Add')
        print(ngraph_bridge.get_disabled_ops())
        c = a + b
        d = tf.placeholder(tf.int32, shape=(5,))
        e = d - c

        def run_test(sess):
            return sess.run((e,),
                            feed_dict={
                                a: np.ones((5,)),
                                d: np.ones((5,))
                            })[0]

        res1 = self.without_ngraph(run_test)
        old_stdout = sys.stdout
        # TODO: Generate unique file name so that no existing file is overwritten
        with open('temp_dump.txt', 'w') as f:
            sys.stdout = f
            res2 = self.with_ngraph(run_test)
            assert (res1 == res2).all()
        sys.stdout = old_stdout
        with open('temp_dump.txt', 'r') as f:
            pass
            # TODO: not working. temp_dump is empty

        # TODO remove temp_dump
        os.environ.pop('NGRAPH_TF_LOG_PLACEMENT', None)
        if log_placement is not None:
            os.environ['NGRAPH_TF_LOG_PLACEMENT'] = log_placement
