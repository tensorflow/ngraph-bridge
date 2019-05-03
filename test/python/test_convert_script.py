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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import os

#TODO fix this
import sys
base_dir = '/localdisk/sarkars/workspace1/ngraph_bridge_tf/dir_1_apr22_grappler/ngraph-bridge'
sys.path.append(base_dir)
sys.path.append(base_dir + '/tools')

from tools.build_utils import command_executor
from tools.convert import convert

from common import NgraphTest


class TestConversionScript(NgraphTest):

    # utility function to make sure input format and location match
    @staticmethod
    def format_and_loc_match(format, loc):
        assert format in ['pbtxt', 'savedmodel']
        implies = lambda p, q: (not p) or (p and q)
        #if its pbtxt, file name has pbtxt AND if its savedmodel, file name does not have pbtxt
        return implies(
            format == 'pbtxt', 'pbtxt' == loc.split('.')[-1]) and implies(
                format == 'savedmodel', 'pbtxt' != loc.split('.')[-1])

    # TODO Certain input combos are commented out (output format = savedmodel)
    @pytest.mark.parametrize(('commandline'), (True, False))
    @pytest.mark.parametrize(('inp_format', 'inp_loc'),
                             (('pbtxt', 'sample_graph.pbtxt'),
                              ('savedmodel', 'sample_graph')))
    @pytest.mark.parametrize(
        ('out_format', 'out_loc'), (('pbtxt', 'sample_graph_modified.pbtxt'),)
    )  #TODO enable (('savedmodel', 'sample_graph_modified'))
    def test_command_line_api(self, inp_format, inp_loc, out_format, out_loc,
                              commandline):
        print('-' * 50)
        print(inp_format, inp_loc, out_format, out_loc, commandline)
        print('-' * 50)
        assert TestConversionScript.format_and_loc_match(inp_format, inp_loc)
        assert TestConversionScript.format_and_loc_match(out_format, out_loc)
        if commandline:
            command_executor('python ' + base_dir +
                             '/tools/convert.py --input' + inp_format + ' ' +
                             inp_loc + ' --outnodes out_node --output' +
                             out_format + ' ' + out_loc)
        else:
            convert(inp_format, inp_loc, out_format, out_loc, ['out_node'])

        #TODO: load the modified graph and run it
        os.remove(out_loc)
