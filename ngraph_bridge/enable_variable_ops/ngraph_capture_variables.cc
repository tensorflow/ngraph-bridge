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

#include <functional>

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"

#include "ngraph_bridge/enable_variable_ops/ngraph_catalog.h"
#include "ngraph_bridge/enable_variable_ops/ngraph_replace_op_utilities.h"
#include "ngraph_bridge/ngraph_api.h"
#include "ngraph_bridge/ngraph_capture_variables.h"
#include "ngraph_bridge/ngraph_utils.h"

using namespace std;
using namespace std::placeholders;

namespace tensorflow {

namespace ngraph_bridge {

//
// Utility function to check if placement on the NGRAPH device has been
// requested.
//
// FIXME(amprocte): stubbed out for now because NGRAPH device is gone.
//
static bool NGraphPlacementRequested(const Node* node) { return true; }

//
// Main entry point for the variable-capture.
//
Status CaptureVariables(Graph* graph, std::set<string> skip_these_nodes) {
  auto ReplaceVariableWithIdentityInfo =
      std::bind(ReplaceVariable, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10,
                skip_these_nodes);
  // This map is no longer static, since we need to do a partial application on
  // ReplaceVariable. Note it is static in RewriteForTracking, since there the
  // partial application is done with a constant argument that will not change
  const std::map<
      const string,
      const pair<
          string,
          function<Status(
              Graph * graph, Node * node, Node * *replacement,
              const string replacement_node_name,
              const string replacement_op_type, const bool just_looking,
              const bool is_tf_just_looking, const bool outputs_ng_supported,
              const int graph_id, const bool is_backend_set)>>>
      CAPTURE_REPLACE_OP_MAP{
          {"ApplyGradientDescent", std::make_pair("NGraphApplyGradientDescent",
                                                  ReplaceApplyGradientDescent)},
          {"Assign", std::make_pair("NGraphAssign", ReplaceAssign)},
          {"AssignAdd", std::make_pair("NGraphAssignAdd", ReplaceAssign)},
          {"AssignSub", std::make_pair("NGraphAssignSub", ReplaceAssign)},
          {"VariableV2",
           std::make_pair("NGraphVariable", ReplaceVariableWithIdentityInfo)}};

  std::vector<Node*> nodes_to_capture;

  for (auto node : graph->op_nodes()) {
    std::set<Node*> ref_list;
    if (NGraphPlacementRequested(node)) {
      // Check if the node is a VariableV2
      if (node->type_string() == "VariableV2") {
        NGRAPH_VLOG(4) << "Found Variable: " << node->name();
        // Add the Variable node to the ref list
        ref_list.insert(node);

        // go over all the nodes leading from VariableV2 and store them
        // in a list if they are ref type
        TF_RETURN_IF_ERROR(StoreRefTypeOutputs(node, &ref_list));

        if (ref_list.size()) {
          for (auto n : ref_list) {
            auto itr = CAPTURE_REPLACE_OP_MAP.find(n->type_string());
            if (itr != CAPTURE_REPLACE_OP_MAP.end()) {
              nodes_to_capture.push_back(n);
            }
          }
          ref_list.clear();
        } else {
// validate_shape is false
// In case of grappler check if we have already captured a variable of this name
#if (NGRAPH_TF_USE_GRAPPLER_OPTIMIZER)
          auto tmp_pair =
              NGraphCatalog::HasTFVarBeenReplacedBefore(node->name());
          if (get<0>(tmp_pair)) {
            // return errors::Internal(node->name(),
            //                       "(VariableV2) was captured in an earlier "
            //                     "graph, but in the current graph ngraph "
            //                   "was unable to capture it");
          }
#endif
        }
      }
    }
  }

  for (auto node : nodes_to_capture) {
    Node* replacement;
    auto itr = CAPTURE_REPLACE_OP_MAP.find(node->type_string());
    // Create the replacement node
    TF_RETURN_IF_ERROR((itr->second.second)(graph, node, &replacement,
                                            node->name(), itr->second.first,
                                            true, false, false, 0, false));
    NGRAPH_VLOG(4) << "Replacing Node " << node->DebugString() << " with "
                   << replacement->DebugString();

    TF_RETURN_IF_ERROR(ReplaceInputControlEdges(graph, node, replacement));
    TF_RETURN_IF_ERROR(ReplaceOutputEdges(graph, node, replacement));
  }  // end of looping through nodes in the capture list

  for (auto node : nodes_to_capture) {
    NGRAPH_VLOG(4) << "Removing: " << node->name();
    graph->RemoveNode(node);
  }

  return Status::OK();
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
