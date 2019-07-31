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
"""nGraph TensorFlow bridge cast operation test

"""
import tensorflow as tf
import numpy as np
import ngraph_bridge
import os

# def test_cast_1d (self):
#     val = tf.placeholder(tf.float32, shape=(2,))
#     out = tf.cast(val, dtype=tf.int32)

#         def run_test(sess):
#             return sess.run(out, feed_dict={val: (5.5, 2.0)})

#         assert (
#             self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()

# def test_cast_2d(self):
#     test_input = ((1.5, 2.5, 3.5), (4.5, 5.5, 6.5))
#     val = tf.placeholder(tf.float32, shape=(2, 3))
#     out = tf.cast(val, dtype=tf.int32)

#     def run_test(sess):
#             return sess.run(out, feed_dict={val: test_input})

#         assert (
#             self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()


def tf_model():
    x = tf.placeholder(tf.float32, shape=(2,), name='x')
    x = tf.cast(x, dtype=tf.bfloat16)
    #testing Cast op
    m = tf.cast(x, dtype=tf.int32)
    m = tf.cast(m, dtype=tf.float32)
    return m, x


def ng_model():
    x = tf.placeholder(tf.float32, shape=(2,), name='x')
    #testing Cast op
    m = tf.cast(x, dtype=tf.int32)
    return m, x


config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False,
    inter_op_parallelism_threads=1)

k_np = np.random.rand(2,)

#Test 1: tf_model TF-native
with tf.Session(config=config) as sess_tf:
    ngraph_bridge.disable()
    tf_out, input = tf_model()
    feed_dict = {input: k_np}
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
    feed_dict = {input: k_np}
    ng_outval = sess_ng.run(ng_out, feed_dict=feed_dict)
    print("Ngraph with NNP backend: ")
    print(ng_outval.dtype, ng_outval[0])

try:
    assert np.allclose(tf_outval, ng_outval)
    print(" \033[92m PASS \033[0m ")
except:
    print(" \033[91m FAIL \033[0m ")
