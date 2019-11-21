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

#include "gtest/gtest.h"

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"

#include "logging/tf_graph_writer.h"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/runtime/interpreter/int_backend.hpp"
#include "ngraph_bridge/ngraph_mark_for_clustering.h"
#include "ngraph_bridge/ngraph_utils.h"
#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

// Inherit INTEREPTER Backend class and override it's is_supported API
class ModifiedInterpreter : public ngraph::runtime::interpreter::INTBackend {
 public:
  bool is_supported(const Node& node) {
    std::cout << "CPU Backend: is_supported called for " << node.name() << "\n";
    if (node.name().find("Reshape") != std::string::npos ||
        node.name().find("Add") != std::string::npos) {
      std::cout << "CPU Backend: is_supported false for " << node.name()
                << "\n";
      return false;
    }
    std::cout << "CPU Backend: is_supported TRUE for " << node.name() << "\n";
    return true;
  }
};

TEST(ModifiedInterpreter, SimpleTest) {
  // Object of the inherited INTERPRETER class
  // IsSupported obj = IsSupported();
  Graph g(OpRegistry::Global());
  auto backend = ngraph::runtime::Backend::create("INTERPRETER");
  // ModifiedInterpreter obj = ModifiedInterpreter(interpreter);

  Tensor t_input_0(DT_FLOAT, TensorShape{2, 3});
  Tensor t_input_1(DT_FLOAT, TensorShape{2, 3});

  Node* node1;
  ASSERT_OK(NodeBuilder("node1", "Const")
                .Attr("dtype", DT_FLOAT)
                .Attr("value", t_input_0)
                .Finalize(&g, &node1));

  Node* node2;
  ASSERT_OK(NodeBuilder("node2", "Const")
                .Attr("dtype", DT_FLOAT)
                .Attr("value", t_input_1)
                .Finalize(&g, &node2));

  Node* node3;
  ASSERT_OK(NodeBuilder("node3", "Add")
                .Input(node1, 0)
                .Input(node2, 0)
                .Attr("T", DT_FLOAT)
                .Finalize(&g, &node3));

  bool is_supported;

  std::map<std::string, std::vector<shared_ptr<ng::Node>>> TFtoNgraphOpMap{
      {"Reshape", {std::make_shared<ngraph::op::Reshape>()}},
      {"Const", {std::make_shared<ngraph::op::Reshape>()}},
      {"Add", {std::make_shared<ngraph::op::Add>()}},
  };

  ASSERT_OK(
      IsSupportedByBackend(node3, backend, TFtoNgraphOpMap, is_supported));
}
}
}
}
