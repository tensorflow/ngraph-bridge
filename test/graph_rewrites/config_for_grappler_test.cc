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

#include "../test_utilities.h"
#include "gtest/gtest.h"
#include "ngraph_assign_clusters.h"
#include "ngraph_backend_manager.h"
#include "ngraph_backend_config.h"
#include "ngraph_mark_for_clustering.h"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tf_graph_writer.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

#define ASSERT_OK(x) ASSERT_EQ((x), ::tensorflow::Status::OK());


TEST(GrapplerConfig, SimpleGraph1) {
  Scope root = Scope::NewRootScope();
  auto A = ops::Const(root.WithOpName("A"), {3.f, 2.f});
  auto B = ops::Const(root.WithOpName("B"), {3.f, 2.f});
  auto Add = ops::Add(root.WithOpName("Add"), A, B);
  auto C = ops::Const(root.WithOpName("C"), {3.f, 2.f});
  auto Mul = ops::Mul(root.WithOpName("Mul"), Add, C);


  Graph graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(&graph));

  // set device specification
  for (auto node : graph.op_nodes()) {
    node->set_requested_device("CPU");
  }

  grappler::GrapplerItem item;
  graph.ToGraphDef(&item.graph);
  ConfigProto config_proto;
  auto backend_name = AttrValue();
  backend_name.set_s("NNPI");
  auto device_id = AttrValue();
  device_id.set_s("1");
  auto num_ice_cores = AttrValue();
  num_ice_cores.set_s("4");
  auto max_batch_size = AttrValue();
  max_batch_size.set_s("64");
  auto& rewriter_config =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();
  rewriter_config.add_optimizers("ngraph-optimizer");
  rewriter_config.set_min_graph_nodes(-1);
  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::ONE);
  auto* custom_config = rewriter_config.add_custom_optimizers();
  custom_config->set_name("ngraph-optimizer");
  (*custom_config->mutable_parameter_map())["ngraph_backend"] = backend_name;
  (*custom_config->mutable_parameter_map())["_ngraph_device_id"] = device_id;
  (*custom_config->mutable_parameter_map())["_ngraph_ice_cores"] = num_ice_cores;
  (*custom_config->mutable_parameter_map())["_ngraph_max_batch_size"] = max_batch_size;

  tensorflow::grappler::MetaOptimizer optimizer(nullptr, config_proto);
  GraphDef output;
  const Status status = optimizer.Optimize(nullptr, item, &output);
  ASSERT_OK(status);

  for (auto node : graph.op_nodes()) {
    auto node_name = node->name();
    cout << "node name " << node->name();
    cout << "\nnode type "<< node->type_string();}
//     if (node_name == "VarX")
//       ASSERT_EQ("NGraphVariable", node->type_string());
//     else if (node_name == "VarY")
//       ASSERT_NE("NGraphVariable", node->type_string());
//     else if (node_name == "AssignX")
//       ASSERT_EQ("NGraphAssign", node->type_string());
//     else if (node_name == "AssignY")
//       ASSERT_NE("NGraphAssign", node->type_string());
//   }
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow
