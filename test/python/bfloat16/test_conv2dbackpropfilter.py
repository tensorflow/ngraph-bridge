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
"""nGraph TensorFlow bridge Conv2d operation test

"""
import pytest

import tensorflow as tf
import numpy as np
import os

import ngraph_bridge

# tf.nn.conv2d_backprop_filter(
#     input,
#     filter_sizes,
#     out_backprop,
#     strides,
#     padding,
#     use_cudnn_on_gpu=True,
#     data_format='NHWC',
#     dilations=[1, 1, 1, 1],
#     name=None
# )

padding = ["SAME", "VALID"]
dilation_nchw = [1, 1, 3, 2]
dilation_nhwc = [1, 3, 2, 1]
stride_NCHW = [1, 1, 2, 2]
stride_NHWC = [1, 2, 2, 1]
filter = np.random.rand(3, 3, 2, 2)
input_size_nhwc = [1, 7, 6, 2]
input_size_nchw = [1, 2, 7, 6]
output_size_nhwc = [1, 4, 3, 2]
output_size_nchw = [1, 2, 4, 3]
input_nhwc = tf.placeholder(tf.float32, shape=input_size_nhwc, name='x')
input_nchw = tf.placeholder(tf.float32, shape=input_size_nchw, name='x')
output_nhwc = tf.placeholder(tf.float32, shape=output_size_nhwc)
output_nchw = tf.placeholder(tf.float32, shape=output_size_nchw)

def tf_model():
    x = tf.cast(input_nhwc, dtype=tf.bfloat16)
    m = tf.nn.conv2d_backprop_filter(
        x,
        filter,
        output_nhwc,
        stride_NHWC,
        padding[1],
        data_format="NHWC",
        dilations=dilation_nhwc,
        name="m")
    m = tf.cast(m, dtype=tf.float32)
    return m, input_nhwc, output_nhwc

def ng_model():
    m = tf.nn.conv2d_backprop_filter(
        input_nchw,
        filter,
        output_nchw,
        stride_NCHW,
        padding[1],
        data_format="NCHW",
        dilations=dilation_nchw,
        name="m")
    return m, input_nchw, output_nchw


config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False,
    inter_op_parallelism_threads=1)

n_np = np.random.rand(1, 2, 7, 6).astype('f')
#Tensorflow supports only NHWC, change input shapes from NCHW to NHWC
t_np = np.transpose(n_np, (0, 2, 3, 1))

n_op = np.random.rand(1, 2, 4, 3).astype('f')
t_op = np.transpose(n_op, (0, 2, 3, 1))

def test_conv2d():
    #Test 1: tf_model TF-native
    with tf.Session(config=config) as sess_tf:
        ngraph_bridge.disable()
        tf_out, input, outprop = tf_model()
        feed_dict = {input: t_np, outprop: t_op }
        tf_outval = sess_tf.run(tf_out, feed_dict=feed_dict)
        print("Native TF: ")
        print(tf_outval.dtype, tf_outval[0])

    #Test 2: model2 with ngraph, NNP backend
    with tf.Session(config=config) as sess_ng:
        ngraph_bridge.enable()
        ngraph_bridge.update_config(config)
        os.environ['NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS'] = '1'
        os.environ['NGRAPH_TF_BACKEND'] = 'NNP'
        ng_out, input, outprop = ng_model()
        feed_dict = {input: n_np, outprop : n_op}
        ng_outval = sess_ng.run(ng_out, feed_dict=feed_dict)
        print("Ngraph with NNP backend: ")
        print(ng_outval.dtype, ng_outval[0])

    # import pdb
    # pdb.set_trace()
    assert np.allclose(
        np.transpose(tf_outval, (0, 3, 2, 1)), ng_outval, rtol=0, atol=1e-02)
