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
"""nGraph TensorFlow variable_update + static input
Var 
| \ 
|   \*
|   Encap 
|   /
Assign (or removed)
* The input to Encap is a static input from Variable
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import getpass
import ctypes

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
import json

import ngraph_bridge
import os

# If the below graph is run for many iterations
# The NGraphVar's NGTensor is updated every iteration
# NGraphVar's TFTensor is not updated as no TF Node needs it
# However StaticInputs are derived from the input TF Tensor (which is stale)
# giving functionally incorrect results
# TF MeanOp expects a static input
#
#    Const     NGraphVar     Const
#      \       /   |   \     /
#      _\|  *|/_   |   _\| |/_
#         Mean     |    Add
#                  |     /
#                 \|/  |/_
#                NGraphAssign
#
# After Encapsulation
#
#            NGraphVar
#             /   |   \
#          *|/_   |   _\|
#     NGEncap1    |   NGEncap2
#                 |     /
#                \|/  |/_
#             NGraphAssign
#


# Define the Graph
def create_graph():
    # Var is initialized by var_init
    var = tf.get_variable('var', [1], dtype=tf.int32)
    var_init = tf.constant([-2])
    var_initialize = var.assign(var_init)

    # Computation of mean
    input1 = tf.constant([[1.0, 2.0], [3.0, 4.0]], name='input1')
    mean = tf.reduce_mean(input1, var)
    const_var = tf.constant([1])
    var_add = tf.add(var, const_var)
    var_update = var.assign(var_add)

    with tf.control_dependencies([var_update]):
        update_op = tf.no_op('train_op')
    return var, var_initialize, mean, update_op


def create_session():
    # Configure the session
    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        inter_op_parallelism_threads=1,
        graph_options=tf.GraphOptions(
            optimizer_options=tf.OptimizerOptions(
                opt_level=tf.OptimizerOptions.L0,
                do_common_subexpression_elimination=False,
                do_constant_folding=False,
                do_function_inlining=False,
            )))
    config_ngraph_enabled = ngraph_bridge.update_config(config)
    sess = tf.Session(config=config_ngraph_enabled)
    return sess


# Compute on NGRAPH
print(" ON NGRAPH")
os.environ['NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS'] = '1'
var, var_init, mean, update_ctrl = create_graph()
ng_sess = create_session()
print("Python: Running with Session")
var_init_value = ng_sess.run((var_init))
print("Var Init Value ", var_init_value)
for i in range(4):
    (result_mean, result_up) = ng_sess.run((mean, update_ctrl))
    print(i)
    print("mean ", result_mean)
print("Final value: ", var.eval(ng_sess))

# Reset the Graph
tf.reset_default_graph()

# # Create session and run on TF
print(" ON TF")
ngraph_bridge.disable()
var, var_init, mean, update_ctrl = create_graph()
tf_sess = create_session()
print("Python: Running with Session")
var_init_value = tf_sess.run((var_init))
print("Var Init Value ", var_init_value)
for i in range(4):
    (result_mean, result_up) = tf_sess.run((mean, update_ctrl))
    print(i)
    print("mean ", result_mean)
print("Final value: ", var.eval(tf_sess))