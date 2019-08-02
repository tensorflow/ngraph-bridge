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
"""nGraph TensorFlow bridge Conv2d operation test

"""
import pytest

import tensorflow as tf
import numpy as np
import os

import ngraph_bridge

#Test Ngraph Op Convolution, TF Op:conv2d
# Implemented based on NNP's unit test TEST(test_assign_layout, convolution_special_case)

np.random.seed(5)

# Colvolution Op is placed on NNP and conerted to bfloat16 only for the special case below, otherwise it falls back to CPU for compute
# Check to assure:
# The input rank is 4-D
# The stride is less than the filter size
# The Window and Data dilation is {1,1}
# Filter shape is allowed
# If any fail, then we should place Op on CPU for compute

#Inputs
N = 1
H = 3
W = 5
C = 1

filter = np.random.rand(1, 1, 1, 2)
input_size_nhwc = [N, H, W, C]
input_size_nchw = [N, C, H, W]
input_nhwc = tf.placeholder(tf.float32, shape=input_size_nhwc, name='x')
input_nchw = tf.placeholder(tf.float32, shape=input_size_nchw, name='x')

n_np = np.random.rand(1, 1, 3, 5).astype('f')
#Tensorflow supports only NHWC, change input shapes from NCHW to NHWC
t_np = np.transpose(n_np, (0, 2, 3, 1))


#TF graph
def tf_model():
    stride_nhwc = [1, 2, 2, 1]
    x = tf.cast(input_nhwc, dtype=tf.bfloat16)
    m = tf.nn.conv2d(
        x, filter, stride_nhwc, "SAME", data_format="NHWC", name="m")
    m = tf.cast(m, dtype=tf.float32)
    return m, input_nhwc


#Ngraph graph
def ng_model():
    stride_nchw = [1, 1, 2, 2]
    m = tf.nn.conv2d(
        input_nchw, filter, stride_nchw, "SAME", data_format="NCHW", name="m")
    return m, input_nchw


config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False,
    inter_op_parallelism_threads=1)


def test_conv2d():
    #Test 1: tf_model TF-native
    with tf.Session(config=config) as sess_tf:
        ngraph_bridge.disable()
        tf_out, input = tf_model()
        feed_dict = {input: t_np}
        tf_outval = sess_tf.run(tf_out, feed_dict=feed_dict)
        print("Native TF: ")
        print(tf_outval.dtype, tf_outval[0])

    #Test 2: model2 with ngraph, NNP backend
    with tf.Session(config=config) as sess_ng:
        ngraph_bridge.enable()
        ngraph_bridge.update_config(config)
        os.environ['NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS'] = '1'
        os.environ['NGRAPH_TF_BACKEND'] = 'NNP'
        ng_out, input = ng_model()
        feed_dict = {input: n_np}
        ng_outval = sess_ng.run(ng_out, feed_dict=feed_dict)
        print("Ngraph with NNP backend: ")
        print(ng_outval.dtype, ng_outval[0])

    assert np.allclose(
        np.transpose(tf_outval, (0, 3, 1, 2)), ng_outval, rtol=0, atol=1e-02)
