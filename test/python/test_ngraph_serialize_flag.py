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
"""Pytest for a simple run on model testing framework

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import platform
import os

import tensorflow as tf
import numpy as np

from common import NgraphTest
import ngraph_bridge

class TestNgraphSerialize(NgraphTest):

    def test_ng_serialize_to_json(self):
        initial_contents = set(os.listdir())
        xshape = (3, 4, 5)
        x = tf.placeholder(tf.float32, shape=xshape)
        out = tf.nn.l2_loss(x)
        values = np.random.rand(*xshape)
        os.environ['NGRAPH_ENABLE_SERIALIZE'] = '1'
        sess_fn = lambda sess: sess.run((out), feed_dict={x: values})
        os.environ.pop('NGRAPH_ENABLE_SERIALIZE', None)
        final_contents = set(os.listdir())
        assert(len(final_contents) - len(initial_contents) == 1)
        expected_file_name = 'tf_function_.--ngraph_cluster_0.json'
        assert(final_contents.difference(initial_contents) == {expected_file_name})
        os.remove(expected_file_name)
        

