# ==============================================================================
#  Copyright 2018 Intel Corporation
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
"""nGraph TensorFlow bridge fusedMatMul tests.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import array_ops
from common import NgraphTest
from tensorflow.python.framework import dtypes

import numpy as np

from google.protobuf import text_format


def get_tensor(graph, tname):
    return graph.get_tensor_by_name("import/" + tname)


def import_pbtxt(pb_filename):
    graph_def = tf.GraphDef()
    with open(pb_filename, "r") as f:
        text_format.Merge(f.read(), graph_def)

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    return graph


class TestFusedMatMul(NgraphTest):
    # TODO: add tests for relu6 as well
    @pytest.mark.parametrize(
        ("filename",),
        (
            ('fusedmatmul_0.pbtxt',),  # Relu
            ('fusedmatmul_1.pbtxt',),  # Relu6
            ('fusedmatmul_2.pbtxt',),  # No activation
        ))
    @pytest.mark.parametrize(("dim1", "dim2", "dim3"), ((3, 2, 2), (3, 4, 5)))
    def test_fusedmatmul_bias_pbtxt(self, filename, dim1, dim2, dim3):
        graph = import_pbtxt(filename)
        with graph.as_default() as g:
            x = get_tensor(g, "Placeholder_3:0")
            y = get_tensor(g, "Placeholder_4:0")
            z = get_tensor(g, "Placeholder_5:0")
            a = get_tensor(g, "Relu_1:0")

            inp1_values = 10 * np.random.rand(dim1, dim2) - 5
            inp2_values = 10 * np.random.rand(dim2, dim3) - 5
            bias_values = 10 * np.random.rand(dim3) - 5

            def run_test(sess):
                return sess.run(a, {
                    x: inp1_values,
                    y: inp2_values,
                    z: bias_values,
                })

            assert np.allclose(
                self.without_ngraph(run_test), self.with_ngraph(run_test))
