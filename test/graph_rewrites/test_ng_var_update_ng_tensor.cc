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

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/public/session.h"

#include "gtest/gtest.h"
#include "logging/tf_graph_writer.h"
#include "ngraph_bridge/enable_variable_ops/ngraph_var.h"
#include "ngraph_bridge/enable_variable_ops/ngraph_variable_update_ng_tensor_op.h"
#include "ngraph_bridge/ngraph_encapsulate_op.h"
#include "ngraph_bridge/ngraph_rewrite_for_tracking.h"
#include "ngraph_bridge/ngraph_utils.h"
#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

TEST(NgVarUpdateNgTensor, SimpleGraph1) {

  Graph g(OpRegistry::Global());
  PartialTensorShape varShape({2, 2});

  Node* var_node;
  ASSERT_OK(NodeBuilder("var_node", "NGraphVariable")
                .Attr("shape", varShape)
                .Attr("dtype", DT_FLOAT)
                .Attr("just_looking", false)
                .Attr("copy_to_tf", false)
                .Attr("container", "")
                .Attr("shared_name", "node1")
                .Attr("ngraph_graph_id", 1)
                .Attr("_ngraph_backend","CPU")
                .Finalize(&g, &var_node));

  std::vector<DataType> input_types;
  input_types.push_back(DT_FLOAT);
  std::vector<DataType> output_types;
  output_types.push_back(DT_FLOAT);
  std::vector<NodeBuilder::NodeOut> inputs;
  inputs.push_back(NodeBuilder::NodeOut(var_node, 0));
  Node* encap_node;
  ASSERT_OK(NodeBuilder("encap_node", "NGraphEncapsulate")
                .Attr("Targuments", input_types)
                .Attr("Tresults", output_types)
                .Attr("ngraph_cluster", 1)
                .Attr("ngraph_graph_id", 1)
                .Attr("ngraph_backend", "CPU")
                .Attr("ngraph_device_id", "1")
                .Input(inputs)
                .Finalize(&g, &encap_node));

  NodeBuilder::NodeOut input_val = NodeBuilder::NodeOut(encap_node, 0);
  Node* assign;
  ASSERT_OK(NodeBuilder("assign", "Assign")
                .Input(var_node)
                .Input(input_val)
                .Attr("T", DT_FLOAT)
                .Finalize(&g, &assign));

  Node* source = g.source_node();
  Node* sink = g.sink_node();
  g.AddEdge(source, Graph::kControlSlot, var_node, Graph::kControlSlot);
  g.AddEdge(assign, Graph::kControlSlot, sink, Graph::kControlSlot);

  ASSERT_OK(RewriteForTracking(&g, 0));

  map<string, Node*> node_map;
  for (auto node : g.op_nodes()) {
    node_map[node->name()] = node;
  }
  ASSERT_EQ(node_map.find("sync_node")->second->type_string(),
            "NGraphVariableUpdateNGTensor");
  node_map.clear();

}
}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow

