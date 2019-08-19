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
"""nGraph TensorFlow bridge test for tf2ngraph script for precompilation with shape hints

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import os
import numpy as np
import shutil
import tensorflow as tf
import ngraph_bridge

from tools.build_utils import command_executor
from tools.tf2ngraph import convert, get_gdef

from common import NgraphTest

def create_graph(p0_shape, p1_shape):
    temp_pbtxt_name = 'temp_graph_'+str(p0_shape)+'__'+str(p1_shape) + '.pbtxt'
    with tf.session() as sess:
        x = tf.placeholder(tf.float32, shape=p0_shape, 'x')
        y = tf.placeholder(tf.float32, shape=p1_shape, 'y')
        z = tf.add(x, y, name= "z")
        tf.io.write_graph(sess.graph, temp_pbtxt_name, name, as_text=True)
    return x, y, z

def get_inputs(p_shape):
    return np.random.rand(*p_shape)

def run_pbtxt(pbtxt_filename, inp0, inp1):
    pass

def helper(p0_shape, p1_shape):
    inp0 = get_inputs(p0_shape)
    inp1 = get_inputs(p1_shape)
    x, y, z, temp_pbtxt_name = create_graph(p0_shape, p1_shape)
    '''
    python tf2ngraph.py --input_pbtxt ../test/test_axpy.pbtxt --output_nodes add --output_pbtxt axpy_ngraph.pbtxt --ng_backend INTERPRETER --shape_hints sample_shape_hints.json --precompile
    python tf2ngraph.py --input_pbtxt temp_pbtxt_name --output_nodes z --output_pbtxt ???.pbtxt --ng_backend INTERPRETER --shape_hints ???.json --precompile
    '''
    # produce input p0_shape, p1_shape (in the caller of helper)
    # produce hints.
    # run tf2ngraph
    # assert the dumped pbtxt has compiled functions
    # Run temp_pbtxt_name on TF. tf_out_val = run_pbtxt(temp_pbtxt_name, inp0, inp1)
    # run out pbtxt. ng_out_val = run_pbtxt(?, inp0, inp1)
    # compare tf_out_val vs ng_out_val
    # delete temp_pbtxt_name


# TODO: Finish this pytest <<<<<<<
class Testtf2ngraphShapehints(NgraphTest):

    def test_tf2ngraph_with_shape_hints(self):
        pass