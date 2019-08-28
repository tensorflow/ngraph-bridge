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
"""nGraph TensorFlow bridge floor operation test
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import platform

import tensorflow as tf
import numpy as np
from common import NgraphTest
import os

class TestDumpingGraphs(NgraphTest):

    def test(self):
        os.environ['NGRAPH_ENABLE_SERIALIZE'] = '1'
        os.environ['NGRAPH_TF_LOG_PLACEMENT'] = '1'
        graph = self.import_pbtxt('flib_graph_1.pbtxt')
        with graph.as_default() as g:
            x = self.get_tensor(g, "Placeholder:0", True)
            y = self.get_tensor(g, "Placeholder_1:0", True)
            z = self.get_tensor(g, "Placeholder_2:0", True)

            a = self.get_tensor(g, "add_1:0", True)
            b = self.get_tensor(g, "Sigmoid:0", True)

            sess_fn = lambda sess: sess.run(
                [a, b], feed_dict={i: np.full((2, 3), 1.0) for i in [x, y, z]})
            
            xx = self.without_ngraph(sess_fn)
            print(xx)

