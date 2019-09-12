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


# TODO move this to tools in the logs parsing functions there
def parse_str(log_str):
    info_dct = {}
    lines = log_str.split('\n')
    # Only select lines with TF_to_NG:
    lines = filter(lambda x: 'TF_to_NG:' in x, lines)
    for ln in lines:
        # Each line is of the form:
        # TF_to_NG: Relu6_ngraph/_0 --> Constant_679, Relu_680, Minimum_681
        splitvals = ln.split('TF_to_NG:')[1].split('-->')
        info_dct[splitvals[0].strip()] = [
            i.split('_')[0] for i in splitvals[1].strip().split(',')
        ]
    return info_dct


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
    expected = {
        'ArithmeticOptimizer/ReplaceMulWithSquare_mul': ['Multiply'],
        'Relu6_ngraph/_0': ['Constant', ' Relu', ' Minimum'],
        'add': ['Add'],
        'add_1': ['Add'],
        'add_1/y': ['Constant'],
        'mul_1': ['Multiply'],
        'mul_1/x': ['Constant']
    }
    assert parse_str(output.decode()) == expected
