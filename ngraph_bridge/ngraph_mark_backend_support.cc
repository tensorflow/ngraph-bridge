/*******************************************************************************
 * Copyright 2020 Intel Corporation
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
    std::map<std::string, bool>& result_map, std::set<ShapeHintMap> hints) {
  result_map.clear();

  unique_ptr<Graph> graph_ptr(new Graph(OpRegistry::Global()));
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(opts, *g, graph_ptr.get()));

  string input_node_type = "Placeholder";
  // map between node name and the PartialShape it contains
  std::map<std::string, PartialShape> node_partial_shape_map =
      GetShapesFromTFInputnodes(graph_ptr.get(), input_node_type);

  if (!node_partial_shape_map.empty()) {
    // If no shape hints are provided but the placeholders contain complete
    // shape. Hence, adding the shapes from placeholders as hints.
    if (hints.size() == 0) {
      NGRAPH_VLOG(5) << "Using shapes from placeholders as hint";

      ShapeHintMap shape_from_placeholders_as_hints;
      for (auto itr : node_partial_shape_map) {
        shape_from_placeholders_as_hints.insert(
            {itr.first, itr.second.get_shape_vector()});
      }
      hints.insert(shape_from_placeholders_as_hints);
    }
  }

  ShapeHintMap inputs_node_shapes_for_compilation;
  bool use_conservative_map = true;
  // Iterate over each shape hint and see if they can be used
  for (ShapeHintMap single_hint : hints) {
    Status status = CanCombineNodeInfoAndHint(
        graph_ptr.get(), input_node_type, node_partial_shape_map, single_hint,
        inputs_node_shapes_for_compilation);
    if (status.ok()) {
      use_conservative_map = false;
      // At this point we have collected all the information and now we are
      // ready to translate
      for (auto node : graph_ptr.get()->op_nodes()) {
        if (node->type_string() == "NGraphEncapsulate") {
          // Check inputs of the encapsulates. They can only be fed by fully
          // concrete shapes (after going through the shape hints) or consts
          std::vector<int32> st_inputs;
          GetStaticInputs(node, &st_inputs);
          // Current assumption is that only encapsulates without static
          // inputs are AOT
          if (st_inputs.size() != 0) {
            return errors::Internal(
                "Found an encapsulate with static inputs, but "
                "that is not supported");
          }

          // get backend.
          std::string backend_name;
          TF_RETURN_IF_ERROR(
              GetNodeAttr(node->attrs(), "ngraph_backend", &backend_name));
          std::string device_id;
          TF_RETURN_IF_ERROR(
              GetNodeAttr(node->attrs(), "ngraph_device_id", &device_id));
          string op_backend_name;
          try {
            op_backend_name = BackendManager::GetBackendCreationString(
                backend_name, device_id);
          } catch (const std::exception& exp) {
            return errors::Internal(
                "Caught exception while creating backend string ", exp.what(),
                "\n");
          }
          TF_RETURN_IF_ERROR(BackendManager::CreateBackend(
              op_backend_name));  // Created a backend here. must free it

          // Backend has been created and setup. Now translate
          string signature;
          std::shared_ptr<ngraph::Function> ng_function;
          TF_RETURN_IF_ERROR(
              PerformTranslation(node, inputs_node_shapes_for_compilation,
                                 signature, ng_function));

          ng::runtime::Backend* op_backend = nullptr;
          try {
            op_backend = BackendManager::GetBackend(op_backend_name);
          } catch (const std::out_of_range& e) {
            NGRAPH_VLOG(5) << "Exception: " << e.what();
            BackendManager::ReleaseBackend(op_backend_name);
            throw;
          }

          TF_RETURN_IF_ERROR(GetBackendSupportInfoForNgfunction(
              op_backend, ng_function, result_map));
          BackendManager::ReleaseBackend(op_backend_name);
        }
      }
    } else {
      use_conservative_map = true;
      break;
    }
  }
  if (use_conservative_map) {
    // If we cannot call TranslateGraph, we do the conservative thing and use
    // the static, hand-populated map
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
  bool is_supported;
  std::map<std::string, bool> ng_result_map;
  for (auto n : ng_function->get_ops()) {
    is_supported = op_backend->is_supported(*n);
    ng_result_map.insert({n->get_name(), is_supported});
  }
  // Get the TF node support information
  TF_RETURN_IF_ERROR(
      GetTFNodeSupportInfo(ng_function, result_map, ng_result_map));
  return Status::OK();
}

Status GetTFNodeSupportInfo(const shared_ptr<ng::Function>& ng_function,
                            std::map<std::string, bool>& result_map,
                            const std::map<std::string, bool>& ng_result_map) {
  bool is_supported;
  for (auto n : ng_function->get_ops()) {
    auto itr = ng_result_map.find(n->get_name());
    is_supported = itr->second;
    std::string tf_node_name = n->get_friendly_name();
    // Look for tf node name in the result map
    // If not found, insert it with the corresponding value for is_supported
    // If found, then check the new value for is_supported
    // If the new value is false, then change the old value to false
    // If the new value is true, then leave it unchanged.
    auto it = result_map.find(tf_node_name);
    if (it != result_map.end()) {
      if (is_supported == false) {
        it->second = false;
      }
    } else {
      result_map.insert({tf_node_name, is_supported});
    }
  }
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