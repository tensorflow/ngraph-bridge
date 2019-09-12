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
"""nGraph TensorFlow bridge TF to NG conversion logs test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np
import tensorflow as tf
import sys
import os
from subprocess import Popen, PIPE
import ngraph_bridge


def test_logging_placement_output():

    log_placement = os.environ.pop('NGRAPH_TF_LOG_PLACEMENT', None)

    os.environ['NGRAPH_TF_LOG_PLACEMENT'] = '1'
    p = Popen(['python', 'simple_script.py'],
              stdin=PIPE,
              stdout=PIPE,
              stderr=PIPE)
    output, err = p.communicate()
    rc = p.returncode
    assert rc == 0

    os.environ.pop('NGRAPH_TF_LOG_PLACEMENT', None)

    if log_placement is not None:
        os.environ['NGRAPH_TF_LOG_PLACEMENT'] = \
            log_placement

    assert output.decode(
    ) == '\n=============New sub-graph logs=============\nNGTF_SUMMARY: Op_not_supported: None\nNGTF_SUMMARY: Op_failed_confirmation: None\nNGTF_SUMMARY: Op_failed_type_constraint: None\nEncapsulate i->j: non contraction reason histogram (Cannot be UNSUPPORTED, NOTANOP or SAMECLUSTER because unsupported ops will not be assigned an encapsulate)\n\nNGTF_SUMMARY: Summary of reasons why a pair of edge connected encapsulates did not merge\nNGTF_SUMMARY: DEADNESS: 0, BACKEND: 0, STATICINPUT: 0, PATHEXISTS: 0\nNGTF_SUMMARY: Summary of reasons why a pair of edge connected clusters did not merge\nNGTF_SUMMARY: NOTANOP: 5, UNSUPPORTED: 3, DEADNESS: 0, BACKEND: 0, SAMECLUSTER: 6, STATICINPUT: 0, PATHEXISTS: 0\n\nNGTF_SUMMARY: Number of nodes in the graph: 11\nNGTF_SUMMARY: Number of nodes marked for clustering: 7 (63% of total nodes)\nNGTF_SUMMARY: Number of nodes assigned a cluster: 7 (63% of total nodes) \t (100% of nodes marked for clustering) \t\nNGTF_SUMMARY: Number of ngraph clusters :1\nNGTF_SUMMARY: Nodes per cluster: 7\nNGTF_SUMMARY: Size of nGraph Cluster[0]:\t7\nNGTF_SUMMARY: Op_deassigned: None\n\nOP_placement:\tHost\t_SOURCE (NoOp)\nOP_placement:\tHost\t_SINK (NoOp)\nOP_placement:\tHost\tPlaceholder (Placeholder)\nOP_placement:\tHost\tRelu6 (IdentityN)\nOP_placement:\tnGraph[0]\tadd_1/y (Const)\nOP_placement:\tnGraph[0]\tmul_1/x (Const)\nOP_placement:\tnGraph[0]\tArithmeticOptimizer/ReplaceMulWithSquare_mul (Square)\nOP_placement:\tnGraph[0]\tmul_1 (Mul)\nOP_placement:\tnGraph[0]\tadd (Add)\nOP_placement:\tnGraph[0]\tadd_1 (Add)\nOP_placement:\tnGraph[0]\tRelu6_ngraph/_0 (Relu6)\n\nNGTF_SUMMARY: Types of edges:: args: 1, retvals: 2, both arg and retval: 0, free: 5, encapsulated: 6, total: 14, computed total: 14\n[array([[4., 4., 4.],\n       [4., 4., 4.]], dtype=float32)]\n\n=============Ending sub-graph logs=============\nTF_to_NG: ArithmeticOptimizer/ReplaceMulWithSquare_mul --> Multiply_673\nTF_to_NG: Relu6_ngraph/_0 --> Constant_679, Relu_680, Minimum_681\nTF_to_NG: add --> Add_677\nTF_to_NG: add_1 --> Add_678\nTF_to_NG: add_1/y --> Constant_671\nTF_to_NG: mul_1 --> Multiply_675\nTF_to_NG: mul_1/x --> Constant_672\n'
