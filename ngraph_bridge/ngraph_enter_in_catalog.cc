/*******************************************************************************
 * Copyright 2019 Intel Corporation
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

#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/serializer.hpp"

#include "logging/ngraph_log.h"
#include "ngraph_bridge/ngraph_catalog.h"
#include "ngraph_bridge/ngraph_enter_in_catalog.h"
#include "ngraph_bridge/ngraph_utils.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

// 1. Populate the PrefetchedInputIndexMap
// 2. Attach Graph Ids to the node

// We collect the below information for the catalog
// 1. If the input to a node (generally the IteratorGetNext)
// is coming from the IteratorV2 node, we catalog it
// i.e. we add the node_name and the input indexes of it's
// outputs to the PrefetchedInputIndexMap for the encap
// op's utilization
// We add mapping of {graphId_nodename : (input_indexs)} to the
// PrefetchedInputIndexMap
//

Status EnterInCatalog(Graph* graph, int graph_id) {
  std::set<Node*> add_to_prefetch_map;
  for (auto node : graph->op_nodes()) {
    // Go over all the inputs of the node
    for (auto edge : node->in_edges()) {
      // If any input is coming from "IteratorV2"
      if (edge->src()->type_string() == "IteratorV2") {
        NGRAPH_VLOG(2) << "src node " << DebugNode(edge->src());
        NGRAPH_VLOG(2) << "adding node " << DebugNode(node)
                       << " to PrefetchedInputIndexMap ";
        add_to_prefetch_map.insert(node);
      }
    }
  }
  for (auto node : add_to_prefetch_map) {
    bool add_to_map = false;
    unordered_set<int> in_indexes_for_encap;
    for (auto edge : node->out_edges()) {
      // Now check that all outputs of this node go to an encap
      // if yes, then catalog it along with the input indexes
      // for the encap op
      if (edge->dst()->type_string() == "NGraphEncapsulate") {
        add_to_map = true;
        in_indexes_for_encap.insert(edge->dst_input());
      } else {
        return errors::Internal("Output of node: ", DebugNode(node),
                                "does not go to the encap op.\n");
      }
    }
    if (add_to_map) {
      try {
        NGraphCatalog::AddToPrefetchedInputIndexMap(graph_id, node->name(),
                                                    in_indexes_for_encap);
      } catch (const std::exception& exp) {
        return errors::Internal("Caught exception while entering in catalog: ",
                                exp.what(), "\n");
      }
    }
  }
  NGRAPH_VLOG(4) << "Entered in Catalog";
  return Status::OK();
}  // enter in catalog

}  // namespace ngraph_bridge

}  // namespace tensorflow
