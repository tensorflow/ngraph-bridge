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

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/public/session.h"

#include "logging/tf_graph_writer.h"
#include "ngraph_bridge/ngraph_api.h"
#include "ngraph_bridge/ngraph_capture_variables.h"
#include "ngraph_bridge/ngraph_utils.h"
#include "ngraph_bridge/version.h"
#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {
Status LoadGraphFromPbTxt(const string& pb_file, Graph* input_graph) {
  // Read the graph
  tensorflow::GraphDef graph_def;
  auto load_graph_status = ReadTextProto(Env::Default(), pb_file, &graph_def);
  if (!load_graph_status.ok()) {
    return load_graph_status;
  }

  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  auto status = ConvertGraphDefToGraph(opts, graph_def, input_graph);
  return status;
}

TEST(CapturePrefetchTest, SmallGraph1) {
  list<string> env_vars{"NGRAPH_TF_USE_PREFETCH"};
  const unordered_map<string, string>& env_map = StoreEnv(env_vars);
  SetEnvVariable("NGRAPH_TF_USE_PREFETCH", "1");
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  Graph input_graph(OpRegistry::Global());

  // Now read the graph
  // test_capture_prefetch.pbtxt was created by running the following
  // command:
  // NGRAPH_TF_DUMP_GRAPHS=1 python examples/axpy_pipelined.py
  // and using the precapture_0000.pbtxt
  ASSERT_OK(LoadGraphFromPbTxt("test_capture_prefetch.pbtxt", &input_graph));
  std::set<string> skip_these_nodes = {};
  ASSERT_OK(CaptureVariables(&input_graph, skip_these_nodes));
  int count_ng_prefetch = 0;
  int count_tf_prefetch = 0;
  for (auto node : input_graph.op_nodes()) {
    if (node->type_string() == "NGraphPrefetchDataset") {
      count_ng_prefetch = count_ng_prefetch + 1;
    }
    if (node->type_string() == "PrefetchDataset") {
      count_tf_prefetch = count_tf_prefetch + 1;
    }
  }
  // There should only be one NGraphPrefetchDataset node
  ASSERT_EQ(count_ng_prefetch, 1);
  // There should be 0 PrefetchDataset nodes as it has been replaced
  ASSERT_EQ(count_tf_prefetch, 0);

  UnsetEnvVariable("NGRAPH_TF_NGVARIABLE_BUFFER_SHARING");
  RestoreEnv(env_map);
}

TEST(CapturePrefetchTest, SmallGraph2) {
  list<string> env_vars{"NGRAPH_TF_USE_PREFETCH"};
  const unordered_map<string, string>& env_map = StoreEnv(env_vars);
  SetEnvVariable("NGRAPH_TF_USE_PREFETCH", "1");
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  Graph input_graph(OpRegistry::Global());

  // Now read the graph
  // test_capture_prefetch_1.pbtxt was created by running resnet50
  // and using the encapsulated_0002.pbtxt
  ASSERT_OK(LoadGraphFromPbTxt("test_capture_prefetch_1.pbtxt", &input_graph));
  std::set<string> skip_these_nodes = {};
  ASSERT_OK(CaptureVariables(&input_graph, skip_these_nodes));
  int count_ng_prefetch = 0;
  int count_tf_prefetch = 0;
  for (auto node : input_graph.op_nodes()) {
    if (node->type_string() == "NGraphPrefetchDataset") {
      count_ng_prefetch = count_ng_prefetch + 1;
      // This NGraphPrefetchDataset node should have only one output
      ASSERT_EQ(node->num_outputs(), 1);
      Node* dst;
      for (auto edge : node->out_edges()) {
        dst = edge->dst();
      }
      // The only one output of NGraphPrefetchDataset should go to
      // a MakeIterator node
      ASSERT_EQ(dst->type_string(), "MakeIterator");
    }
    if (node->type_string() == "PrefetchDataset") {
      count_tf_prefetch = count_tf_prefetch + 1;
    }
  }
  // There should only be one NGraphPrefetchDataset node
  ASSERT_EQ(count_ng_prefetch, 1);
  // There should be 2 PrefetchDataset nodes that were not to be captured
  ASSERT_EQ(count_tf_prefetch, 2);

  UnsetEnvVariable("NGRAPH_TF_NGVARIABLE_BUFFER_SHARING");
  RestoreEnv(env_map);
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow