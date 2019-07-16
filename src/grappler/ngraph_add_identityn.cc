/*******************************************************************************
 * Copyright 2017-2019 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/
#include "ngraph_add_identityn.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

Status AddIdentityN(Graph* input_graph, std::set<string> skip_these_nodes) {
  for (auto node : input_graph->op_nodes()) {
    bool fetch_node = false;
    fetch_node = skip_these_nodes.find(node->name()) != skip_these_nodes.end();
    std::set<string> new_nodes_to_be_skipped;
    if (fetch_node) {
      NGRAPH_VLOG(5) << "NGTF_OPTIMIZER: Fetch Node " << node->name();
      // Check the number of outputs of the 'fetch_node'
      // Only move further to create an IdentityN node
      // if it is greater than 0

      // TODO: check and remove this comment::
      // Also, make sure that none of the output types is
      // a ref type because IdentityN does not support
      // an input of type ref type
      if (node->num_outputs()) {
        std::vector<NodeBuilder::NodeOut> inputs;
        std::vector<DataType> input_types;
        for (int i = 0; i < node->num_outputs(); i++) {
          // DT_X_REF = DT_X + 100. kDataTypeRefOffset = 100 here
          // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.proto
          input_types.push_back(static_cast<DataType>(
              node->output_type(i) -
              (IsRefType(node->output_type(i)) ? kDataTypeRefOffset : 0)));
          inputs.push_back(NodeBuilder::NodeOut(node, i));
        }

        NGRAPH_VLOG(5) << "NGTF_OPTIMIZER: Creating an IdentityN node";
        Node* identityN_node;
        TF_RETURN_IF_ERROR(NodeBuilder(node->name(), "IdentityN")
                               .Attr("T", input_types)
                               .Input(inputs)
                               .Device(node->assigned_device_name())
                               .Finalize(input_graph, &identityN_node));

        identityN_node->set_assigned_device_name(node->assigned_device_name());

        // Rename the skip node
        // Get a new name for the node with the given prefix
        // We will use the 'original-node-name_ngraph' as the prefix
        string new_name = input_graph->NewName(node->name() + "_ngraph");
        // TODO: Use (guaranteed) unique name here
        node->set_name(new_name);
        NGRAPH_VLOG(5) << "NGTF_OPTIMIZER: New name for fetch node "
                       << node->name();
      } else {
        NGRAPH_VLOG(5) << "NGTF_OPTIMIZER: num outputs " << node->num_outputs();
        NGRAPH_VLOG(5) << "NGTF_OPTIMIZER: Cannot construct an IdentityN node";
      }
    }
  }
  return Status::OK();
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
