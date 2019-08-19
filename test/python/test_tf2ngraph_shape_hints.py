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
import json

from tools.build_utils import command_executor
from tools.tf2ngraph import convert, get_gdef

from common import NgraphTest


def create_graph(p0_shape, p1_shape):
    temp_pbtxt_name = 'temp_graph_in_' + str(p0_shape) + '__' + str(
        p1_shape) + '.pbtxt'
    with tf.session() as sess:
        x = tf.placeholder(tf.float32, shape=p0_shape, name='x')
        y = tf.placeholder(tf.float32, shape=p1_shape, name='y')
        z = tf.add(x, y, name="z")
        tf.io.write_graph(sess.graph, temp_pbtxt_name, name, as_text=True)
    return x, y, z


def get_inputs(p_shape):
    return np.random.rand(*p_shape)


def run_pbtxt(pbtxt_filename, inp0, inp1):
    pass


def check_pbtxt_has_exec(pbtxt_filename):
    with open(pbtxt_filename, 'w') as f:
        contents = '\n'.join(f.readlines())
        assert contents.count('_ngraph_aot_requested') == 1
        # TODO add the shape signature to the 2 asserts below
        assert contents.count('_ngraph_aot_ngexec_') == 1
        assert contents.count('_ngraph_aot_ngfunction_') == 1


def helper(p0_shape, p1_shape, shapehints):
    inp0 = get_inputs(p0_shape)
    inp1 = get_inputs(p1_shape)
    x, y, z, temp_in_pbtxt_name = create_graph(p0_shape, p1_shape)
    temp_out_pbtxt_name = 'temp_graph_out_' + str(p0_shape) + '__' + str(
        p1_shape) + '.pbtxt'
    json_name = 'temp_shape_hints.json'
    # shapehints is a list of dictionaries (keys are node names, vals are lists (of shapes))
    json.dump({"shape_hints": shapehints}, open(json_name, 'w'))
    '''
    TODO: remove this command line comment
    python tf2ngraph.py --input_pbtxt ../test/test_axpy.pbtxt --output_nodes add --output_pbtxt axpy_ngraph.pbtxt --ng_backend INTERPRETER --shape_hints sample_shape_hints.json --precompile
    python tf2ngraph.py --input_pbtxt temp_pbtxt_name --output_nodes z --output_pbtxt out_pbtxt_name --ng_backend INTERPRETER --shape_hints json_name --precompile
    '''

    command_executor('python ../../tools/tf2ngraph.py --input_pbtxt ' +
                     temp_in_pbtxt_name + ' --output_nodes z --output_pbtxt ' +
                     temp_out_pbtxt_name + ' --ng_backend INTERPRETER ' +
                     ' --shape_hints ' + json_name + ' --precompile')

    check_pbtxt_has_exec(temp_out_pbtxt_name)

    tf_out_val = run_pbtxt(temp_in_pbtxt_name, inp0, inp1)
    ng_out_vals = run_pbtxt(temp_out_pbtxt_name, inp0, inp1)

    # TODO: compare tf_out_val vs ng_out_vals

    os.remove(temp_in_pbtxt_name)
    os.remove(temp_out_pbtxt_name)
    os.remove(json_name)


# TODO: Finish this pytest <<<<<<<
class Testtf2ngraphShapehints(NgraphTest):

    def test_tf2ngraph_with_shape_hints(self):
        # parameterize with input shapes and shape hints and call helper
        pass
