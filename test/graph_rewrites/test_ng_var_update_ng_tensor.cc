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

#include <vector>

#include "gtest/gtest.h"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"

#include "logging/tf_graph_writer.h"
#include "ngraph_bridge/enable_variable_ops/ngraph_var.h"
#include "ngraph_bridge/enable_variable_ops/ngraph_variable_update_ng_tensor_op.h"
#include "ngraph_bridge/ngraph_rewrite_for_tracking.h"
#include "ngraph_bridge/ngraph_utils.h"
#include "test/test_utilities.h"
#include "test/tf_fake_input.h"

namespace tensorflow {
namespace ngraph_bridge {
namespace testing {

TEST(NGVarUpdateNGTensorOpTest, SimpleGraph1) {
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
                .Attr("_ngraph_backend", "CPU")
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
  ASSERT_EQ(node_map.find("var_node_sync_node")->second->type_string(),
            "NGraphVariableUpdateNGTensor");

  Node *in_0, *in_ctrl, *sync_node = node_map.at("var_node_sync_node");
  // NOTE:node->input_edge(...), node->input_node(...) cannot be used for
  // control edges
  int edge_count = 0;
  for (auto edge : sync_node->in_edges()) {
    if (edge->dst_input() == 0) {
      in_0 = edge->src();
      ASSERT_TRUE(IsRefType(sync_node->input_type(0)));
    } else if (edge->dst_input() == Graph::kControlSlot) {
      in_ctrl = edge->src();
    }
    edge_count++;
  }

  // Assert on edges connected to sync node
  ASSERT_EQ(edge_count, 2);
  ASSERT_EQ(in_0, node_map.at("var_node/non_ng_outputs/gid_0"));
  ASSERT_EQ(in_ctrl, node_map.at("assign"));
  ASSERT_EQ(sync_node->num_outputs(), 1);

  for (auto edge : sync_node->out_edges()) {
    if ((edge != nullptr)) {
      ASSERT_EQ(edge->dst()->IsRetval(), true);
    }
  }

  node_map.clear();
}  // end SimpleGraph1

TEST(NGVarUpdateNGTensorOpTest, SimpleGraph2) {
  list<string> env_vars{"NGRAPH_TF_NGVARIABLE_BUFFER_SHARING"};
  const unordered_map<string, string>& env_map = StoreEnv(env_vars);
  SetEnvVariable("NGRAPH_TF_NGVARIABLE_BUFFER_SHARING", "0");

  Scope root = Scope::NewRootScope();

  PartialTensorShape varShape({2, 2});
  auto var = ops::Variable(root.WithOpName("Var"), varShape, DT_FLOAT);
  auto init_value = ops::Const(root, {{1.f, 1.f}, {1.f, 1.f}});
  auto var_assign = ops::Assign(root.WithOpName("Assign1"), var, init_value);

  auto accum = ops::Variable(root.WithOpName("accum"), varShape, DT_FLOAT);
  auto init_value2 = ops::Const(root, {{3.f, 3.f}, {3.f, 3.f}});
  auto accum_assign =
      ops::Assign(root.WithOpName("Assign2"), accum, init_value2);

  auto grad = ops::Const(root, {{2.f, 2.f}, {2.f, 2.f}});

  auto lr = ops::Const(root, 1.f);

  ops::ApplyAdagrad::Attrs use_locking;
  use_locking = use_locking.UseLocking(true);
  auto applyadagrad_t = ops::ApplyAdagrad(root.WithOpName("Adagrad"), var,
                                          accum, lr, grad, use_locking);

  // Turn off optimizations so that all the nodes are processed
  tensorflow::SessionOptions options;
  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(tensorflow::OptimizerOptions_Level_L0);
  options.config.mutable_graph_options()
      ->mutable_rewrite_options()
      ->set_constant_folding(tensorflow::RewriterConfig::OFF);

  // Run on nGraph
  ActivateNGraph();
  ClientSession ng_session(root, options);
  std::vector<tensorflow::Tensor> ng_outputs1;
  std::vector<tensorflow::Tensor> ng_outputs2;
  ASSERT_OK(ng_session.Run({{var_assign, accum_assign}}, &ng_outputs1));

  ASSERT_OK(ng_session.Run({applyadagrad_t}, &ng_outputs2));

  DeactivateNGraph();

  // Run on TF
  ClientSession tf_session(root, options);
  std::vector<tensorflow::Tensor> tf_outputs1;
  std::vector<tensorflow::Tensor> tf_outputs2;
  ASSERT_OK(tf_session.Run({{var_assign, accum_assign}}, &tf_outputs1));

  ASSERT_OK(tf_session.Run({applyadagrad_t}, &tf_outputs2));

  Compare(tf_outputs1, ng_outputs1);
  Compare(tf_outputs2, ng_outputs2);

  ActivateNGraph();
  UnsetEnvVariable("NGRAPH_TF_NGVARIABLE_BUFFER_SHARING");
  RestoreEnv(env_map);
}  // SimpleGraph2

}  // testing
}  // ngraph_bridge
}  // tensorflow