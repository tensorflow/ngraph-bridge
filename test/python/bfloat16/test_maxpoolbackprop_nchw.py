# ==============================================================================
#  Copyright 2018-2019 Intel Corporation
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
"""nGraph TensorFlow bridge MaxPoolBackprop operation test

"""

# Currently, this test fails with a segmentation fault
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np
import os

import tensorflow as tf
from tensorflow.python.ops.gen_nn_ops import max_pool_grad

import ngraph_bridge

N = 4
H = 8
W = 8
C = 3

N1 = 4
H1 = 3
W1 = 8
C1 = 1

N2 = 4
H2 = 4
W2 = 8
C2 = 2

grad_nhwc = {
    "VALID": np.random.rand(N1, C1, H1, W1).astype('f'),
    "SAME": np.random.rand(N2, C2, H2, W2).astype('f')
}

strides = [1, 2, 2, 1]
ksize = [1, 3, 3, 1]

def tf_model(padding):
    orig_in = tf.placeholder(tf.float32, shape=[N, C, H, W])
    orig_in_c = tf.cast(orig_in, tf.bfloat16)
    orig_in_t = tf.transpose(orig_in_c, (0, 2, 3, 1))
    orig_out = tf.placeholder(tf.float32, shape=[N, C, H, W])
    orig_out_c = tf.cast(orig_out, tf.bfloat16)
    orig_out_t = tf.transpose(orig_out_c, (0, 2, 3, 1))
    if padding == "VALID":
        grad = tf.placeholder(tf.float32, shape=(N1, C1, H1, W1))
    elif padding == "SAME":
        grad = tf.placeholder(tf.float32, shape=(N2, C2, H2, W2))
    grad_c = tf.cast(grad, tf.bfloat16)
    grad_t = tf.transpose(grad_c, (0, 2, 3, 1))
    out = max_pool_grad(orig_in_t, orig_out_t, grad_t, ksize, strides, padding=padding, data_format="NHWC")
    output = tf.cast(out, tf.float32)
    output_t = tf.transpose(output, (0, 3, 1, 2))
    return output_t, orig_in, orig_out, grad

def ng_model(padding):
    orig_in = tf.placeholder(tf.float32, shape=[N, C, H, W])
    orig_out = tf.placeholder(tf.float32, shape=[N, C, H, W])
    if padding == "VALID":
        grad = tf.placeholder(tf.float32, shape=(N1, C1, H1, W1))
    elif padding == "SAME":
        grad = tf.placeholder(tf.float32, shape=(N2, C2, H2, W2))
    out = max_pool_grad(orig_in, orig_out, grad, ksize, strides, padding=padding, data_format="NHWC")
    return out, orig_in, orig_out, grad

config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False,
    inter_op_parallelism_threads=1)

i_np = np.random.rand(N, C, H, W).astype('f') # NHWC
o_np = np.random.rand(N, C, H, W).astype('f') # NHWC

@pytest.mark.parametrize("padding", ("VALID",))
def test_maxpoolbackprop_nhwc(padding):
    np_nhwc = grad_nhwc[padding]

    #Test 1: tf_model TF-native
    with tf.Session(config=config) as sess_tf:
        ngraph_bridge.disable()
        tf_out, orig_in, orig_out, grad = tf_model(padding)
        feed_dict = {orig_in: i_np, orig_out: o_np, grad: np_nhwc}
        tf_outval = sess_tf.run(tf_out, feed_dict=feed_dict)

    #Test 2: model2 with ngraph, NNP backend
    with tf.Session(config=config) as sess_ng:
        ngraph_bridge.enable()
        ngraph_bridge.update_config(config)
        os.environ['NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS'] = '1'
        os.environ['NGRAPH_TF_BACKEND'] = 'NNP'
        ng_out, orig_in, orig_out, grad = ng_model(padding)
        feed_dict = {orig_in: i_np, orig_out: o_np, grad: np_nhwc}
        ng_outval = sess_ng.run(ng_out, feed_dict=feed_dict)

    assert (np.allclose(tf_outval, ng_outval,  rtol=0, atol=1e-02))
