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

import argparse
import pdb
import tensorflow as tf
from google.protobuf import text_format
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.grappler import tf_optimizer
import ngraph_bridge
import os


def run_ngraph_grappler_optimizer(input_gdef, output_nodes):
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(input_gdef, name="")
    grappler_meta_graph_def = tf.train.export_meta_graph(
        graph_def=graph.as_graph_def(add_shapes=True), graph=graph)

    _to_bytes = lambda s: s.encode("utf-8", errors="surrogateescape")
    output_collection = meta_graph_pb2.CollectionDef()
    output_list = output_collection.node_list.value
    for i in output_nodes:
        if isinstance(i, tf.Tensor):
            output_list.append(_to_bytes(i.name))
        else:
            output_list.append(_to_bytes(i))
    # TODO(laigd): use another key as the outputs are really not train_op.
    grappler_meta_graph_def.collection_def["train_op"].CopyFrom(
        output_collection)

    rewriter_config = rewriter_config_pb2.RewriterConfig(
        meta_optimizer_iterations=rewriter_config_pb2.RewriterConfig.ONE,
        custom_optimizers=[
            rewriter_config_pb2.RewriterConfig.CustomGraphOptimizer(
                name="ngraph-optimizer")
        ])

    session_config_with_trt = tf.ConfigProto()
    session_config_with_trt.graph_options.rewrite_options.CopyFrom(
        rewriter_config)
    input_gdef = tf_optimizer.OptimizeGraph(
        session_config_with_trt, grappler_meta_graph_def, graph_id=b"tf_graph")
    return input_gdef


def get_gdef_from_savedmodel(export_dir):
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                                   export_dir)
        return sess.graph.as_graph_def()


def get_gdef_from_pbtxt(filename):
    graph_def = tf.GraphDef()
    with open(filename, "r") as f:
        text_format.Merge(f.read(), graph_def)
    return graph_def


def get_input_gdef(input_dict):
    # input_dict is a dictionary of input formats and locations. all except one should be None
    format, location = filter_dict("input", input_dict)
    return {
        'savedmodel': get_gdef_from_savedmodel,
        'pbtxt': get_gdef_from_pbtxt
    }[format](location)


def prepare_argparser(formats):
    parser = argparse.ArgumentParser()
    in_out_groups = [parser.add_argument_group(i) for i in ['input', 'output']]
    for grp in in_out_groups:
        inp_out_group = grp.add_mutually_exclusive_group()
        for format in formats[grp.title]:
            opt_name = grp.title + format
            inp_out_group.add_argument(
                "--" + opt_name, help="Location of " + grp.title + " " + format)
    # Note: no other option must begin with "input" or "output"
    parser.add_argument(
        "--outnodes", help="Comma separated list of output nodes")
    return parser.parse_args()


def filter_dict(prefix, dictionary):
    assert prefix in ['input', 'output']
    current_format = list(
        filter(lambda x: x.startswith(prefix) and dictionary[x] is not None,
               dictionary))
    assert len(current_format) == 1, "Got " + str(
        len(current_format)) + " input formats, expected only 1"
    # [len(prefix):] deletes the initial "input" in the string
    stripped = current_format[0][len(prefix):]
    assert stripped in allowed_formats[
        prefix], "Got " + prefix + " format = " + stripped + " but only support " + str(
            allowed_formats[prefix])
    return (stripped, dictionary[prefix + stripped])


def save_gdef_to_savedmodel(gdef, location):
    raise Exception("Implement me")


def save_gdef_to_pbtxt(gdef, location):
    tf.io.write_graph(
        gdef,
        os.path.dirname(location),
        os.path.basename(location),
        as_text=True)


def save_model(gdef, out_dict):
    format, location = filter_dict("output", out_dict)
    return {
        'savedmodel': save_gdef_to_savedmodel,
        'pbtxt': save_gdef_to_pbtxt
    }[format](gdef, location)


allowed_formats = {
    "input": ['savedmodel', 'pbtxt'],
    "output": ['savedmodel', 'pbtxt']
}


def main():
    args = prepare_argparser(allowed_formats)
    input_gdef = get_input_gdef(args.__dict__)
    output_gdef = run_ngraph_grappler_optimizer(input_gdef,
                                                args.outnodes.split(','))
    save_model(output_gdef, args.__dict__)
    print('Bye')


if __name__ == '__main__':
    main()
    #python convert.py --inputsavedmodel test_graph_SM --outnodes out_node --outputpbtxt test_graph_SM_mod.pbtxt
    #python convert.py --inputpbtxt test_graph_SM.pbtxt --outnodes out_node --outputpbtxt test_graph_SM_mod.pbtxt
