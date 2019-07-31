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
import tensorflow as tf
import numpy as np
import ngraph_bridge
import os

# tf.nn.conv2d(
#     input,
#     filter=None,
#     strides=None,
#     padding=None,
#     use_cudnn_on_gpu=True,
#     data_format='NHWC',
#     dilations=[1, 1, 1, 1],
#     name=None,
#     filters=None
# )
padding = ["SAME", "VALID"]
dilation_nchw = [1, 1, 3, 2]
dilation_nhwc = [1, 3, 2, 1]
stride_NCHW = [1, 1, 2, 2]
stride_NHWC = [1, 2, 2, 1]
filter = np.random.rand(3, 3, 2, 2)
input_size_nhwc = [1, 7, 6, 2]
input_size_nchw = [1, 2, 7, 6]
input_nhwc = tf.placeholder(tf.float32, shape=input_size_nhwc, name='x')
input_nchw = tf.placeholder(tf.float32, shape=input_size_nchw, name='x')


def tf_model():
    x = tf.cast(input_nhwc, dtype=tf.bfloat16)
    m = tf.nn.conv2d(
        x,
        filter,
        stride_NHWC,
        padding[1],
        data_format="NHWC",
        dilations=dilation_nhwc,
        name="m")
    m = tf.cast(m, dtype=tf.float32)
    return m, input_nhwc


def ng_model():
    m = tf.nn.conv2d(
        input_nchw,
        filter,
        stride_NCHW,
        padding[1],
        data_format="NCHW",
        dilations=dilation_nchw,
        name="m")
    return m, input_nchw


config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False,
    inter_op_parallelism_threads=1)

n_np = np.random.rand(1, 2, 7, 6).astype('f')
t_np = np.transpose(n_np, (0, 2, 3, 1))

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

result = np.allclose(
    np.transpose(tf_outval, (0, 3, 2, 1)), ng_outval, rtol=0, atol=1e-02)

if result:
    print(" \033[92m PASS \033[0m ")
else:
    print(" \033[91m FAIL \033[0m ")
