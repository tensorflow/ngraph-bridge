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

import os
import platform
import random

import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2

from google.protobuf import text_format

import ngraph_bridge

__all__ = ['LIBNGRAPH_BRIDGE', 'NgraphTest']

_ext = 'dylib' if platform.system() == 'Darwin' else 'so'

LIBNGRAPH_BRIDGE = 'libngraph_bridge.' + _ext


class NgraphTest(object):

    @staticmethod
    def get_tensor(graph, tname, loading_from_protobuf):
        return graph.get_tensor_by_name(("", "import/")[loading_from_protobuf] + tname)

    @staticmethod
    def import_protobuf(pb_filename):
        graph_def = tf.GraphDef()
        if pb_filename.endswith("pbtxt"):
            with open(pb_filename, "r") as f:
                text_format.Merge(f.read(), graph_def)
        else:
            with open(pb_filename, "rb") as f:
                graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def)
        return graph

    def with_ngraph(self, l, config=tf.ConfigProto()):
        # TODO: Stop grappler on failure (Add fail_on_optimizer_errors=True)
        if ngraph_bridge.is_grappler_enabled():
            rewrite_options = rewriter_config_pb2.RewriterConfig(
                meta_optimizer_iterations=rewriter_config_pb2.RewriterConfig.
                ONE,
                min_graph_nodes=-1,
                custom_optimizers=[
                    rewriter_config_pb2.RewriterConfig.CustomGraphOptimizer(
                        name="ngraph-optimizer")
                ])
            config = tf.ConfigProto(
                graph_options=tf.GraphOptions(rewrite_options=rewrite_options))

        ngraph_tf_disable_deassign_clusters = os.environ.pop(
            'NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS', None)

        os.environ['NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS'] = '1'
        ngraph_bridge.enable()
        with tf.Session(config=config) as sess:
            retval = l(sess)

        os.environ.pop('NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS', None)

        if ngraph_tf_disable_deassign_clusters is not None:
            os.environ['NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS'] = \
                ngraph_tf_disable_deassign_clusters

        return retval

    def without_ngraph(self, l, config=tf.ConfigProto()):
        ngraph_tf_disable_deassign_clusters = os.environ.pop(
            'NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS', None)

        ngraph_bridge.disable()
        with tf.Session(config=config) as sess:
            retval = l(sess)

        if ngraph_tf_disable_deassign_clusters is not None:
            os.environ['NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS'] = \
                ngraph_tf_disable_deassign_clusters

        return retval

    # returns a vector of length 'vector_length' with random
    # float numbers in range [start,end]
    def generate_random_numbers(self,
                                vector_length,
                                start,
                                end,
                                datatype="DTYPE_FLOAT"):
        if datatype == "DTYPE_INT":
            return [random.randint(start, end) for i in range(vector_length)]
        return [random.uniform(start, end) for i in range(vector_length)]
