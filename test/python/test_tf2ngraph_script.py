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
"""nGraph TensorFlow bridge test for tf2ngraph script for precompilation

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import os
import numpy as np
import shutil
import tensorflow as tf
import ngraph_bridge

from tools.build_utils import command_executor
from tools.tf2ngraph import convert, get_gdef, Tf2ngraphJson, get_gdef_from_protobuf

from common import NgraphTest


class Testtf2ngraph(NgraphTest):

    # utility function to make sure input format and location match
    @staticmethod
    def format_and_loc_match(format, loc):
        assert format in ['pb', 'pbtxt', 'savedmodel']
        implies = lambda p, q: (not p) or (p and q)
        #if its pbtxt, file name has pbtxt AND if its savedmodel, file name does not have pbtxt
        return implies(
            format == 'pbtxt', 'pbtxt' == loc.split('.')[-1]) and implies(
                format == 'savedmodel', 'pbtxt' != loc.split('.')[-1])

    @pytest.mark.parametrize(('commandline'),
                             (True,))  #TODO: add False for functional api test
    @pytest.mark.parametrize(('inp_format', 'inp_loc', 'out_node_name'), (
        ('pbtxt', 'sample_graph.pbtxt', 'out_node'),
        ('savedmodel', 'sample_graph', 'out_node'),
        ('pb', 'sample_graph.pb', 'out_node'),
        ('pbtxt', 'sample_graph_nodevice.pbtxt', 'out_node'),
        ('pbtxt', 'sample_graph_nodevice.pbtxt', None),
    ))
    @pytest.mark.parametrize(('out_format',), (
        ('pbtxt',),
        ('pb',),
        ('savedmodel',),
    ))
    @pytest.mark.parametrize(('ng_device', 'shape_hints', 'precompile'),
                             (('CPU', [], False), ('INTERPRETER', [{}], True),
                              ('INTERPRETER', [], False)))
    # In sample_graph.pbtxt, the input shape is fully specified, so we don't need to pass shape hints for precompile
    def test_command_line_api(self, inp_format, inp_loc, out_node_name,
                              out_format, commandline, ng_device, shape_hints,
                              precompile):
        # Only run this test when grappler is enabled
        if not ngraph_bridge.is_grappler_enabled():
            return
        assert Testtf2ngraph.format_and_loc_match(inp_format, inp_loc)
        out_loc = inp_loc.split('.')[0] + '_modified' + (
            '' if out_format == 'savedmodel' else ('.' + out_format))
        try:
            (shutil.rmtree, os.remove)[os.path.isfile(out_loc)](out_loc)
        except:
            pass
        conversion_successful = False
        try:
            optional_backend_params = {
                'CPU': {
                    'device_config': '0'
                },
                'INTERPRETER': {
                    'test_echo': '1'
                }
            }[ng_device]
            config_file_name = 'temp_config_file.json'
            Tf2ngraphJson.dump_json(config_file_name, optional_backend_params,
                                    shape_hints)
            if commandline:
                # In CI this test is expected to be run out of artifacts/test/python
                # out_node_str is empty if out_node_name is None.
                # Automatic output node inference is supposed to kick in that case.
                # we monkeypatch the builtin function input to simulate an user input
                out_node_str = ' ' if out_node_name is None else ' --output_nodes ' + out_node_name + ' '
                with mock.patch('builtins.input', return_value="out_node"):
                    command_executor(
                        'python ../../tools/tf2ngraph.py --input_' +
                        inp_format + ' ' + inp_loc + out_node_str +
                        ' --output_' + out_format + ' ' + out_loc +
                        ' --ng_backend ' + ng_device + ' --config_file ' +
                        config_file_name + ("", " --precompile ")[precompile])
            else:
                convert(inp_format, inp_loc, out_format, out_loc, ['out_node'],
                        ng_device, optional_backend_params, shape_hints,
                        precompile)
            conversion_successful = True
        finally:
            if not conversion_successful:
                try:
                    (shutil.rmtree, os.remove)[os.path.isfile(out_loc)](out_loc)
                    os.remove(config_file_name)
                except:
                    pass
        assert conversion_successful

        gdef = get_gdef(out_format, out_loc)
        (shutil.rmtree, os.remove)[os.path.isfile(out_loc)](out_loc)
        os.remove(config_file_name)

        with tf.Graph().as_default() as g:
            tf.import_graph_def(gdef, name='')
            # The graph should have exactly one encapsulate
            assert len([
                0 for i in g.get_operations() if i.type == 'NGraphEncapsulate'
            ]) == 1
            # TODO: check that the encapsulate op has correct backend and extra params attached to it
            x = self.get_tensor(g, "x:0", False)
            y = self.get_tensor(g, "y:0", False)
            out = self.get_tensor(g, "out_node:0", False)

            sess_fn = lambda sess: sess.run(
                [out], feed_dict={i: np.zeros((10,)) for i in [x, y]})

            res1 = self.with_ngraph(sess_fn)
            res2 = self.without_ngraph(sess_fn)

            exp = [0.5 * np.ones((10,))]
            # Note both run on Host (because NgraphEncapsulate can only run on host)
            assert np.isclose(res1, res2).all()
            # Comparing with expected value
            assert np.isclose(res1, exp).all()

    def test_output_node_inference_for_saved_model(self):
        # This saved model has input and output specified, and tf2ngraph should be able to infer it without user input
        export_dir = 'temp_pytest_savedmodel'
        out_loc = 'temp_pytest_pbtxt.pbtxt'
        # TODO take care of file deletion
        try:
            shutil.rmtree(export_dir)
            os.remove(out_loc)
        except:
            pass

        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, [None, 784], name="input")
            W = tf.constant(np.zeros([784, 10]), tf.float32)
            b = tf.constant(np.zeros([10]), tf.float32)
            linear = tf.matmul(x, W) + b
            y = tf.nn.softmax(linear, name="output")
            tf.saved_model.simple_save(
                sess,
                export_dir,
                inputs={"inp1": x},
                outputs={
                    "out1": y,
                    "out2": linear
                })

        command_executor('python ../../tools/tf2ngraph.py --input_savedmodel ' +
                         export_dir + ' --output_pbtxt ' + out_loc)

        # Load the tf2ngraph dumped graphdef and find if it has nodes of type IdentityN that are the provided output names in the simple_save line above
        # Note that for grappler output node names are IdentityN, so that is what we are looking for
        gdef = get_gdef_from_protobuf(out_loc)
        found_out1 = False
        found_out2 = False
        for n in gdef.node:
            if n.op == 'IdentityN':
                if n.name == 'output':
                    found_out1 = True
                elif n.name == 'add':
                    found_out2 = True
                else:
                    assert False, "Found node " + n.name + " of type IdentityN. But did not expect this output node name to be inferred by tf2ngraph"
        assert found_out1, "Did not find output name 'output' of type IdentityN which was supposed to be autoinferred"
        assert found_out2, "Did not find output name 'add' of type IdentityN which was supposed to be autoinferred"
        os.remove(out_loc)
        shutil.rmtree(export_dir)
