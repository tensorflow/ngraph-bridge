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

#include "ngraph_bridge/ngraph_mark_backend_support.h"

namespace ng = ngraph;
using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

// TODO use TranslateGraph
Status GetBackendSupportInfoForTFSubgraph(
    const ng::runtime::Backend* op_backend, GraphDef* g,
    std::map<std::string, bool>& result_map) {
  result_map.clear();
  // TODO: fill this function
  // Call translate graph. Then call GetBackendSupportInfoForNgfunction

  // TODO, populate possible_to_translate correctly later
  bool possible_to_translate = false;
  if (possible_to_translate) {
    // call translategraph etc and GetBackendSupportInfoForNgfunction
    // TODO
    return errors::Internal("Unimplemented: Call TranslateGraph");
  } else {
    // If we cannot call TranslateGraph, we do the conservative thing and use
    // the static, hand-populated map
    unique_ptr<Graph> graph_ptr(new Graph(OpRegistry::Global()));
    GraphConstructorOptions opts;
    opts.allow_internal_ops = true;
    TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(opts, *g, graph_ptr.get()));
    bool supported_op;
    for (auto node : graph_ptr->nodes()) {
      if (NodeIsMarkedForClustering(node)) {
        TF_RETURN_IF_ERROR(
            IsSupportedByBackend(node, op_backend, supported_op));
        result_map.insert({node->name(), supported_op});
      }
    }
  }
  return Status::OK();
}

Status GetBackendSupportInfoForNgfunction(
    const ng::runtime::Backend* op_backend,
    const shared_ptr<ng::Function>& ng_function,
    std::map<std::string, bool>& result_map) {
  result_map.clear();
  // TODO: fill this function

  return Status::OK();
}

Status IsSupportedByBackend(
    const Node* node, const ng::runtime::Backend* op_backend,
    const std::map<std::string, std::set<std::shared_ptr<ngraph::Node>>>&
        TFtoNgraphOpMap,
    bool& is_supported) {
  is_supported = true;

  auto ng_op = TFtoNgraphOpMap.find(node->type_string());
  if (ng_op == TFtoNgraphOpMap.end()) {
    return errors::Internal("TF Op is not found in the map: ",
                            node->type_string());
  }

  // Loop through the ngraph op list to query
  for (auto it = ng_op->second.begin(); it != ng_op->second.end(); it++) {
    // Pass ngraph node to check if backend supports this op
    auto ret = op_backend->is_supported(**it);
    if (!ret) {
      is_supported = false;
      return Status::OK();
    }
  }
  return Status::OK();
}

// Check if op is supported by backend using is_supported API
Status IsSupportedByBackend(const Node* node,
                            const ng::runtime::Backend* op_backend,
                            bool& is_supported) {
  return IsSupportedByBackend(node, op_backend, GetTFToNgOpMap(), is_supported);
}

}  // namespace ngraph_bridge
}  // namespace tensorflow