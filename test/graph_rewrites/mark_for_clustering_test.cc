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
#include "ngraph_bridge/ngraph_cluster_manager.h"
#include "ngraph_bridge/ngraph_mark_for_clustering.h"
#include "ngraph_bridge/ngraph_utils.h"
#include "test/dummy_backend.h"
#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

class MarkForClusteringTest : public ::testing::Test {
 protected:
  void SetUp() override {
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

    Node* node4;
    ASSERT_OK(NodeBuilder("node4", "Abs")
                  .Input(node3, 0)
                  .Attr("T", DT_FLOAT)
                  .Finalize(&g, &node4));

    // Add edges from SRC to node1 and node2
    // Add edge from node3 to SINK
    // The graph is disconnected without these edges
    Node* source = g.source_node();
    Node* sink = g.sink_node();
    g.AddEdge(source, Graph::kControlSlot, node1, Graph::kControlSlot);
    g.AddEdge(source, Graph::kControlSlot, node2, Graph::kControlSlot);
    g.AddEdge(node4, Graph::kControlSlot, sink, Graph::kControlSlot);
  }

  size_t NumNodesMarkedForClustering() {
    size_t ctr = 0;
    for (auto node : g.nodes()) {
      ctr += (NodeIsMarkedForClustering(node) ? 1 : 0);
    }
    return ctr;
  }

  Graph g{OpRegistry::Global()};
};

TEST_F(MarkForClusteringTest, QueryBackendForSupportTest1) {
  string current_backend = "dummy";
  ngraph::runtime::dummy::DummyBackend db;

  vector<Node*> nodes_marked_for_clustering = {};

  NGraphClusterManager::EvictAllClusters();

  ASSERT_EQ(NGraphClusterManager::GetNumClusters(), 0);

  ASSERT_OK(QueryBackendForSupport(&g, &db, current_backend, {},
                                   nodes_marked_for_clustering));

  // No nodes were marked for clustering, hence the CM is not expected to be
  // populated
  ASSERT_EQ(NGraphClusterManager::GetNumClusters(), 0);
  ASSERT_EQ(NumNodesMarkedForClustering(), 0);

  NGraphClusterManager::EvictAllClusters();
}

TEST_F(MarkForClusteringTest, QueryBackendForSupportTest2) {
  string current_backend = "dummy";
  ngraph::runtime::dummy::DummyBackend db;

  vector<Node*> nodes_marked_for_clustering;
  for (auto node : g.nodes()) {
    nodes_marked_for_clustering.push_back(node);
  }

  NGraphClusterManager::EvictAllClusters();

  ASSERT_EQ(NGraphClusterManager::GetNumClusters(), 0);

  // We have marked source/sink (all ops for marking). We expect it to fail
  ASSERT_NOT_OK(QueryBackendForSupport(&g, &db, current_backend, {},
                                       nodes_marked_for_clustering));

  // The QueryBackendForSupport failed, so we should not see any markings
  ASSERT_EQ(NGraphClusterManager::GetNumClusters(), 0);
  // Nodes are marked for clustering, but AssignClusters failed, so
  // QueryBackendForSupport exited. But we did not clear up the markings
  ASSERT_EQ(NumNodesMarkedForClustering(), 6);

  NGraphClusterManager::EvictAllClusters();
}

TEST_F(MarkForClusteringTest, QueryBackendForSupportTest3) {
  string current_backend = "dummy";
  ngraph::runtime::dummy::DummyBackend db;

  vector<Node*> nodes_marked_for_clustering;
  for (auto node : g.nodes()) {
    if (node->type_string() == "Const" || node->type_string() == "Add" ||
        node->type_string() == "Abs") {
      nodes_marked_for_clustering.push_back(node);
    }
  }

  NGraphClusterManager::EvictAllClusters();

  ASSERT_EQ(NGraphClusterManager::GetNumClusters(), 0);

  // We have marked source/sink (all ops for marking). We expect it to fail
  ASSERT_OK(QueryBackendForSupport(&g, &db, current_backend, {},
                                   nodes_marked_for_clustering));

  // The dummy backend does not support anything, so nothing should have been
  // marked
  ASSERT_EQ(NGraphClusterManager::GetNumClusters(), 0);
  ASSERT_EQ(NumNodesMarkedForClustering(), 0);

  NGraphClusterManager::EvictAllClusters();
}

TEST(MarkForClustering, SimpleTest) {
  Graph g(OpRegistry::Global());

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

  Node* node4;
  ASSERT_OK(NodeBuilder("node4", "Abs")
                .Input(node3, 0)
                .Attr("T", DT_FLOAT)
                .Finalize(&g, &node4));

  // Add edges from SRC to node1 and node2
  // Add edge from node3 to SINK
  // The graph is disconnected without these edges
  Node* source = g.source_node();
  Node* sink = g.sink_node();
  g.AddEdge(source, Graph::kControlSlot, node1, Graph::kControlSlot);
  g.AddEdge(source, Graph::kControlSlot, node2, Graph::kControlSlot);
  g.AddEdge(node4, Graph::kControlSlot, sink, Graph::kControlSlot);

  const char* ng_backend_env_value = std::getenv("NGRAPH_TF_BACKEND");
  string expected_backend{"CPU"};
  if (ng_backend_env_value != nullptr) {
    expected_backend = std::string(ng_backend_env_value);
  }
  ASSERT_OK(MarkForClustering(&g, {}, expected_backend, {}));

  string backend;
  const set<string> nodes_expected_to_be_marked{"node1", "node2", "node3",
                                                "node4"};
  for (auto node : g.op_nodes()) {
    ASSERT_OK(GetNodeBackend(node, &backend));
    ASSERT_EQ(backend, expected_backend);
    ASSERT_EQ(nodes_expected_to_be_marked.find(node->name()) !=
                  nodes_expected_to_be_marked.end(),
              NodeIsMarkedForClustering(node));
  }

  ResetMarkForClustering(&g);
  for (auto node : g.op_nodes()) {
    ASSERT_NOT_OK(GetNodeBackend(node, &backend));
    ASSERT_FALSE(NodeIsMarkedForClustering(node));
  }
  NGraphClusterManager::EvictAllClusters();
}
}
}
}