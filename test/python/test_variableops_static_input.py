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

from common import NgraphTest


class TestVariableStaticInputs(NgraphTest):
    # Define the Graph
    # def create_graph(self):
    #     # Var is initialized by var_init
    #     var = tf.get_variable('var', [1], dtype=tf.int32)
    #     var_init = tf.constant([-2])
    #     var_initialize = var.assign(var_init)

    #     # Computation of mean
    #     input1 = tf.constant([[1.0, 2.0], [3.0, 4.0]], name='input1')
    #     mean = tf.reduce_mean(input1, var)
    #     const_var = tf.constant([1])
    #     var_add = tf.add(var, const_var)
    #     var_update = var.assign(var_add)

    #     with tf.control_dependencies([var_update]):
    #         update_op = tf.no_op('train_op')
    #     return var, var_initialize, mean, update_op

    def test_variable_static_input(self):

        def run_test(sess):
            # Var is initialized by var_init
            var = tf.get_variable('var', [1], dtype=tf.int32)
            var_init = tf.constant([-2])
            var_initialize = var.assign(var_init)

            # Computation of mean
            input1 = tf.constant([[1.0, 2.0], [3.0, 4.0]], name='input1')
            mean = tf.reduce_mean(input1, var)
            const_var = tf.constant([1])
            var_add = tf.add(var, const_var)

            with tf.control_dependencies([mean]):
                var_update = var.assign(var_add)

            with tf.control_dependencies([var_update]):
                update_op = tf.no_op('train_op')

            var_init_value = sess.run((var_initialize))
            print("Var Init Value ", var_init_value)
            for i in range(3):
                (result_mean, result_up) = sess.run((mean, update_op))
                print(i)
                print("mean ", result_mean)
            var_final_val = var.eval(sess)
            print("Final value: ", var_final_val)
            return var_init_value, result_mean, var_final_val

        #var, var_init, mean, update_ctrl = self.create_graph()
        ng_var_init_val, ng_mean, ng_final = self.with_ngraph(run_test)
        # Reset Graph
        # tf.reset_default_graph()
        # var, var_init, mean, update_ctrl = self.create_graph()
        # It is necessary to reset the graph because of the variables
        #tf_var_init_val, tf_mean, tf_final = self.without_ngraph(run_test)