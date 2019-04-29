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

from common import NgraphTest


class TestOpDisableOperations(NgraphTest):

    @pytest.mark.parametrize(("op_list",), (('Add',), ('Add,Sub',), ('',),
                                            ('_InvalidOp',)))
    def test_disable_op_1(self, op_list):
        ngraph_bridge.set_disabled_ops(op_list)
        assert ngraph_bridge.get_disabled_ops() == op_list.encode("utf-8")
        # Running get_disabled_ops twice to see nothing has changed between 2 consecutive calls
        assert ngraph_bridge.get_disabled_ops() == op_list.encode("utf-8")

    # TODO: this test is not working as expected. need to capture NGRAPH_TF_LOG_PLACEMENT
    def test_disable_2(self, capsys):
        # TODO: make the env var settign and resettign a decorator
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

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()

        #import pdb; pdb.set_trace()
        logs = capsys.readouterr()
        os.environ.pop('NGRAPH_TF_LOG_PLACEMENT', None)
        if log_placement is not None:
            os.environ['NGRAPH_TF_LOG_PLACEMENT'] = log_placement
