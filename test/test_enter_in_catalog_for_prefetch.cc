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
#include "ngraph_bridge/ngraph_assign_clusters.h"
#include "ngraph_bridge/ngraph_capture_variables.h"
#include "ngraph_bridge/ngraph_catalog.h"
#include "ngraph_bridge/ngraph_cluster_manager.h"
#include "ngraph_bridge/ngraph_deassign_clusters.h"
#include "ngraph_bridge/ngraph_encapsulate_clusters.h"
#include "ngraph_bridge/ngraph_enter_in_catalog.h"
#include "ngraph_bridge/ngraph_mark_for_clustering.h"
#include "ngraph_bridge/ngraph_rewrite_for_tracking.h"
#include "ngraph_bridge/ngraph_utils.h"
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

TEST(PrefetchCatalogTest, SmallGraph1) {
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  Graph input_graph(OpRegistry::Global());

  // Now read the graph
  ASSERT_OK(
      LoadGraphFromPbTxt("test_catalog_for_prefetch.pbtxt", &input_graph));

  ASSERT_OK(EnterInCatalog(&input_graph, 0));
  ASSERT_TRUE(
      NGraphCatalog::ExistsInPrefetchedInputIndexMap("0_ngraph_cluster_4"));

  // Clean up
  NGraphCatalog::ClearCatalog();
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow