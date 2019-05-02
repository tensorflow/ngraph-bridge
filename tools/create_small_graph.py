import tensorflow as tf
import numpy as np


def build_graph(output_name):
    with tf.device(
            '/cpu:0'
    ):  #TODO: this line is needed, else grappler pass does not work
        # Create graph
        x = tf.placeholder(dtype=tf.float32, shape=(10,), name='x')
        y = tf.placeholder(dtype=tf.float32, shape=(10,), name='y')
        z = x + 2 * y
        return {'outputs': [tf.sigmoid(z, name=output_name)], 'inputs': [x, y]}


graph_in_out_nodes = build_graph('out_node')

export_dir = 'test_graph_SM'

builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
with tf.Session() as sess:
    res = sess.run(
        graph_in_out_nodes['outputs'],
        feed_dict={k: np.zeros((10,)) for k in graph_in_out_nodes['inputs']})
    print(res)
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.TRAINING])
    builder.add_meta_graph([tf.saved_model.tag_constants.SERVING],
                           strip_default_attrs=True)
builder.save()

tf.io.write_graph(
    sess.graph.as_graph_def(), '.', 'test_graph_SM.pbtxt', as_text=True)
