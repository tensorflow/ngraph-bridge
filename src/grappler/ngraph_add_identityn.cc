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
    bool ref_type = false;
    fetch_node = skip_these_nodes.find(node->name()) != skip_these_nodes.end();
    std::set<string> new_nodes_to_be_skipped;
    if (fetch_node) {
      NGRAPH_VLOG(5) << "NGTF_OPTIMIZER: Fetch Node " << node->name();
      // Check the number of outputs of the 'fetch_node'
      // Only move further to create an IdentityN node
      // if it is greater than 0
      // Also, make sure that none of the output types is
      // a ref type because IdentityN does not support
      // an input of type ref type
      if (node->num_outputs()) {
        std::vector<NodeBuilder::NodeOut> inputs;
        std::vector<DataType> input_types;
        for (int i = 0; i < node->num_outputs(); i++) {
          if (IsRefType(node->output_type(i))) {
            NGRAPH_VLOG(5) << "NGTF_OPTIMIZER: "
                           << "Datatype for the node output"
                           << " at index " << i << " is ref type";
            ref_type = true;
          }
          input_types.push_back(node->output_type(i));
          inputs.push_back(NodeBuilder::NodeOut(node, i));
        }

        if (ref_type) {
          if (node->num_outputs() == 1) {
            NGRAPH_VLOG(5)
              << "NGTF_OPTIMIZER: Constructing Identity nodes instead of IdentityN";
            // DT_X_REF = DT_X + 100. // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.proto
            DataType non_ref = static_cast<DataType>(input_types[0] - kDataTypeRefOffset);
            Node* identity_node;
            TF_RETURN_IF_ERROR(NodeBuilder(node->name(), "Identity")
                                  .Attr("T", non_ref)
                                  .Input(inputs[0])
                                  .Device(node->assigned_device_name())
                                  .Finalize(input_graph, &identity_node));
            identity_node->set_assigned_device_name(node->assigned_device_name());
          } else {
            NGRAPH_VLOG(5)
              << "NGTF_OPTIMIZER: Cannot construct IdentityN or Identity";
          }
        } else {
          NGRAPH_VLOG(5) << "NGTF_OPTIMIZER: Creating an IdentityN node";
          Node* identityN_node;
          TF_RETURN_IF_ERROR(NodeBuilder(node->name(), "IdentityN")
                                .Attr("T", input_types)
                                .Input(inputs)
                                .Device(node->assigned_device_name())
                                .Finalize(input_graph, &identityN_node));

          identityN_node->set_assigned_device_name(node->assigned_device_name());
        }

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
