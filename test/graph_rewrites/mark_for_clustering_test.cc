/*******************************************************************************
 * Copyright 2019-2020 Intel Corporation
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

class MarkForClusteringTestBase : public ::testing::Test {
 protected:
  size_t NumNodesMarkedForClustering() {
    size_t ctr = 0;
    for (auto node : g.nodes()) {
      ctr += (NodeIsMarkedForClustering(node) ? 1 : 0);
    }
    return ctr;
  }

  Graph g{OpRegistry::Global()};
};

class MarkForClusteringTest1 : public MarkForClusteringTestBase {
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
    // Add edge from node4 to SINK
    // The graph is disconnected without these edges
    Node* source = g.source_node();
    Node* sink = g.sink_node();
    g.AddEdge(source, Graph::kControlSlot, node1, Graph::kControlSlot);
    g.AddEdge(source, Graph::kControlSlot, node2, Graph::kControlSlot);
    g.AddEdge(node4, Graph::kControlSlot, sink, Graph::kControlSlot);
  }
};

TEST_F(MarkForClusteringTest1, QueryBackendForSupportTest1) {
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

TEST_F(MarkForClusteringTest1, QueryBackendForSupportTest2) {
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

TEST_F(MarkForClusteringTest1, QueryBackendForSupportTest3) {
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

  ASSERT_OK(QueryBackendForSupport(&g, &db, current_backend, {},
                                   nodes_marked_for_clustering));

  // The dummy backend does not support anything, so nothing should have been
  // marked
  ASSERT_EQ(NGraphClusterManager::GetNumClusters(), 0);
  ASSERT_EQ(NumNodesMarkedForClustering(), 0);

  NGraphClusterManager::EvictAllClusters();
}

TEST_F(MarkForClusteringTest1, QueryBackendForSupportTest4) {
  string current_backend = "dummy";
  ngraph::runtime::dummy::DummyBackend2 db;

  vector<Node*> nodes_marked_for_clustering;
  for (auto node : g.nodes()) {
    if (node->type_string() == "Const" || node->type_string() == "Add" ||
        node->type_string() == "Abs") {
      nodes_marked_for_clustering.push_back(node);
    }
  }

  NGraphClusterManager::EvictAllClusters();

  ASSERT_EQ(NGraphClusterManager::GetNumClusters(), 0);

  ASSERT_OK(QueryBackendForSupport(&g, &db, current_backend, {},
                                   nodes_marked_for_clustering));

  // This dummy backend supports everything
  ASSERT_EQ(NGraphClusterManager::GetNumClusters(), 1);
  ASSERT_EQ(NumNodesMarkedForClustering(), 4);

  NGraphClusterManager::EvictAllClusters();
}

// Deassigning cluster
TEST_F(MarkForClusteringTest1, QueryBackendForSupportTest5) {
  string current_backend = "dummy";
  ngraph::runtime::dummy::DummyBackend2 db;

  vector<Node*> nodes_marked_for_clustering;
  for (auto node : g.nodes()) {
    if (node->type_string() == "Const" || node->type_string() == "Add") {
      nodes_marked_for_clustering.push_back(node);
    }
  }

  NGraphClusterManager::EvictAllClusters();

  ASSERT_EQ(NGraphClusterManager::GetNumClusters(), 0);

  ASSERT_OK(QueryBackendForSupport(&g, &db, current_backend, {},
                                   nodes_marked_for_clustering));

  // This dummy backend supports everything
  // AssignClusters creates a cluster since an initial cluster is proposed but
  // it is deassigned. However the ClusterManager contains an (empty) entry
  // TODO if we make ClusterManager a map instead of a vector, we can get rid of
  // this. Then Deassign pass would clear up ClusterManager
  ASSERT_EQ(NGraphClusterManager::GetNumClusters(), 1);
  // Though const and Add are marked for clustering, nothing gets clustered
  // since the trivial cluster of const->add<-const is deassigned
  ASSERT_EQ(NumNodesMarkedForClustering(), 0);

  NGraphClusterManager::EvictAllClusters();
}

// Mark const, add and abs for clustering, but backend only supports const and
// add. Forms a trivial cluster const->add<-const, which is deassigned
TEST_F(MarkForClusteringTest1, QueryBackendForSupportTest6) {
  string current_backend = "dummy";
  ngraph::runtime::dummy::DummyBackend3 db;
  db.set_supported_behaviour({
      std::make_shared<ngraph::op::Add>(),
      ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{},
                                   {2.0f}),
  });

  vector<Node*> nodes_marked_for_clustering;
  for (auto node : g.nodes()) {
    if (node->type_string() == "Const" || node->type_string() == "Add" ||
        node->type_string() == "Abs") {
      nodes_marked_for_clustering.push_back(node);
    }
  }

  NGraphClusterManager::EvictAllClusters();

  ASSERT_EQ(NGraphClusterManager::GetNumClusters(), 0);

  ASSERT_OK(QueryBackendForSupport(&g, &db, current_backend, {},
                                   nodes_marked_for_clustering));

  // This dummy backend supports Add and const but not abs
  ASSERT_EQ(NGraphClusterManager::GetNumClusters(), 1);
  ASSERT_EQ(NumNodesMarkedForClustering(), 0);

  NGraphClusterManager::EvictAllClusters();
}

// Mark const, add and abs for clustering, but backend only supports abs and
// add. Forms one cluster
TEST_F(MarkForClusteringTest1, QueryBackendForSupportTest7) {
  string current_backend = "dummy";
  ngraph::runtime::dummy::DummyBackend3 db;
  db.set_supported_behaviour({std::make_shared<ngraph::op::Abs>(),
                              std::make_shared<ngraph::op::Add>()});

  vector<Node*> nodes_marked_for_clustering;
  for (auto node : g.nodes()) {
    if (node->type_string() == "Const" || node->type_string() == "Add" ||
        node->type_string() == "Abs") {
      nodes_marked_for_clustering.push_back(node);
    }
  }

  NGraphClusterManager::EvictAllClusters();

  ASSERT_EQ(NGraphClusterManager::GetNumClusters(), 0);

  ASSERT_OK(QueryBackendForSupport(&g, &db, current_backend, {},
                                   nodes_marked_for_clustering));

  // This dummy backend supports Add and abs but not const
  ASSERT_EQ(NGraphClusterManager::GetNumClusters(), 1);
  ASSERT_EQ(NumNodesMarkedForClustering(), 2);

  NGraphClusterManager::EvictAllClusters();
}

class MarkForClusteringTest2 : public MarkForClusteringTestBase {
 protected:
  void SetUp() override {
    Node* node1;
    ASSERT_OK(NodeBuilder("node1", "Placeholder")
                  .Attr("dtype", DT_FLOAT)
                  .Finalize(&g, &node1));

    Node* node2;
    ASSERT_OK(NodeBuilder("node2", "Softplus")
                  .Input(node1, 0)
                  .Attr("T", DT_FLOAT)
                  .Finalize(&g, &node2));

    Node* node3;
    ASSERT_OK(NodeBuilder("node3", "SquaredDifference")
                  .Input(node2, 0)
                  .Input(node2, 0)
                  .Attr("T", DT_FLOAT)
                  .Finalize(&g, &node3));

    Node* node4;
    ASSERT_OK(NodeBuilder("node4", "Abs")
                  .Input(node3, 0)
                  .Attr("T", DT_FLOAT)
                  .Finalize(&g, &node4));

    Node* node5;
    ASSERT_OK(NodeBuilder("node5", "Abs")
                  .Input(node4, 0)
                  .Attr("T", DT_FLOAT)
                  .Finalize(&g, &node5));

    // Add edges from SRC to node1
    // Add edge from node5 to SINK
    // The graph is disconnected without these edges
    Node* source = g.source_node();
    Node* sink = g.sink_node();
    g.AddEdge(source, Graph::kControlSlot, node1, Graph::kControlSlot);
    g.AddEdge(node5, Graph::kControlSlot, sink, Graph::kControlSlot);
  }
};


// Rejection because of partial support:
// In this case some nodes of "softplus" are supported, some not, so softplus
// will not be supported
// All nodes of squareddifference and abs are supported, so they will form a
// cluster
TEST_F(MarkForClusteringTest2, QueryBackendForSupportTest8) {
  string current_backend = "dummy";
  ngraph::runtime::dummy::DummyBackend3 db;
  db.set_supported_behaviour({std::make_shared<ngraph::op::Abs>(),
                              std::make_shared<ngraph::op::Exp>(),
                              std::make_shared<ngraph::op::Subtract>(),
                              std::make_shared<ngraph::op::Multiply>(),
                              std::make_shared<ngraph::op::Broadcast>(),
                              });

  vector<Node*> nodes_marked_for_clustering;
  for (auto node : g.nodes()) {
    if (node->type_string() == "Softplus" ||
        node->type_string() == "SquaredDifference" ||
        node->type_string() == "Abs") {
      nodes_marked_for_clustering.push_back(node);
    }
  }

  NGraphClusterManager::EvictAllClusters();

  ASSERT_EQ(NGraphClusterManager::GetNumClusters(), 0);

  ASSERT_OK(QueryBackendForSupport(&g, &db, current_backend, {},
                                   nodes_marked_for_clustering));

  ASSERT_EQ(NGraphClusterManager::GetNumClusters(), 1);
  ASSERT_EQ(NumNodesMarkedForClustering(), 3);

  NGraphClusterManager::EvictAllClusters();
}


// An example where 2 passes were needed to get the right mark_for_clustering
// Tf node A --> X->Y
// Tf node B-> M->N
// A->B translates to X->Y->M->N
// Y and M are not supported by backend, but after fusion YM is supported (X->YM->N).
// But N is not supported.
// So From X->Y->M->N, we get markings: X(T)->Y(T)->M(T)->N(F)  (T means true (supported) and viceversa)
// but because N is not supported, TF node B will not be marked.
// So we get a ng function X->Y
// But now we have to submit this again to the backend for inspection. Now the backend will say, oh wait, Y does not lead to M, hence I cannot support it. So now both A and B fall back to TF.

// To simulate the above scenario the is_supported of the test dummy backend will say it does not support "Multiply" the first time it is called
// every other ng node is marked supported the first time is_supported is called
// Then squareddifference is not be supported,
// Then we reject squareddifference and only mark softplus for clustering.
// but now in the second time, is_supported will say it does not support log
// Now softplus will be rejected as well
// Only the 2 Abs will be clustered.


TEST_F(MarkForClusteringTest2, QueryBackendForSupportTest9) {
  string current_backend = "dummy";
  ngraph::runtime::dummy::DummyBackend4 db;

  vector<Node*> nodes_marked_for_clustering;
  for (auto node : g.nodes()) {
    if (node->type_string() == "Softplus" ||
        node->type_string() == "SquaredDifference" ||
        node->type_string() == "Abs") {
      nodes_marked_for_clustering.push_back(node);
    }
  }

  NGraphClusterManager::EvictAllClusters();

  ASSERT_EQ(NGraphClusterManager::GetNumClusters(), 0);

  ASSERT_OK(QueryBackendForSupport(&g, &db, current_backend, {},
                                   nodes_marked_for_clustering));

  // One actual cluster and one deassigned (softplus)
  ASSERT_EQ(NGraphClusterManager::GetNumClusters(), 2);
  
  ASSERT_EQ(NumNodesMarkedForClustering(), 2);
  // Only the 2 abs nodes are marked for clustering
  for (auto node : g.nodes()) {
    ASSERT_EQ((node->type_string() == "Abs"), NodeIsMarkedForClustering(node));
  }

  NGraphClusterManager::EvictAllClusters();
}


// Similar to QueryBackendForSupportTest9, but we do not deassign
TEST_F(MarkForClusteringTest2, QueryBackendForSupportTest10) {
  list<string> env_vars{"NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS"};
  const unordered_map<string, string>& env_map = StoreEnv(env_vars);
  SetEnvVariable("NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS", "1");

  string current_backend = "dummy";
  ngraph::runtime::dummy::DummyBackend4 db;

  vector<Node*> nodes_marked_for_clustering;
  for (auto node : g.nodes()) {
    if (node->type_string() == "Softplus" ||
        node->type_string() == "SquaredDifference" ||
        node->type_string() == "Abs") {
      nodes_marked_for_clustering.push_back(node);
    }
  }

  NGraphClusterManager::EvictAllClusters();

  ASSERT_EQ(NGraphClusterManager::GetNumClusters(), 0);

  ASSERT_OK(QueryBackendForSupport(&g, &db, current_backend, {},
                                   nodes_marked_for_clustering));

  // One cluster containing the abs
  ASSERT_EQ(NGraphClusterManager::GetNumClusters(), 1);
  
  ASSERT_EQ(NumNodesMarkedForClustering(), 2);
  // Only the 2 abs nodes are marked for clustering
  for (auto node : g.nodes()) {
    ASSERT_EQ((node->type_string() == "Abs"), NodeIsMarkedForClustering(node));
  }

  NGraphClusterManager::EvictAllClusters();
  UnsetEnvVariable("NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS");
  RestoreEnv(env_map);
}

// Similar to QueryBackendForSupportTest9, but using DummyBackend3
// In this case we expect softplus to be clustered
TEST_F(MarkForClusteringTest2, QueryBackendForSupportTest11) {
  list<string> env_vars{"NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS"};
  const unordered_map<string, string>& env_map = StoreEnv(env_vars);
  SetEnvVariable("NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS", "1");

  auto constant = ngraph::op::Constant::create(ngraph::element::f32,
                                               ngraph::Shape{}, {2.0f});

  string current_backend = "dummy";
  ngraph::runtime::dummy::DummyBackend3 db;
  db.set_supported_behaviour({std::make_shared<ngraph::op::Abs>(),
                              std::make_shared<ngraph::op::Exp>(),
                              std::make_shared<ngraph::op::Log>(),
                              std::make_shared<ngraph::op::Add>(),
                              constant
                              });

  vector<Node*> nodes_marked_for_clustering;
  for (auto node : g.nodes()) {
    if (node->type_string() == "Softplus" ||
        node->type_string() == "SquaredDifference" ||
        node->type_string() == "Abs") {
      nodes_marked_for_clustering.push_back(node);
    }
  }

  NGraphClusterManager::EvictAllClusters();

  ASSERT_EQ(NGraphClusterManager::GetNumClusters(), 0);

  ASSERT_OK(QueryBackendForSupport(&g, &db, current_backend, {},
                                   nodes_marked_for_clustering));

  // One cluster containing the abs and one containing softplus
  ASSERT_EQ(NGraphClusterManager::GetNumClusters(), 2);
  
  ASSERT_EQ(NumNodesMarkedForClustering(), 3);
  // Only the 2 abs nodes and softplus are marked for clustering
  for (auto node : g.nodes()) {
    ASSERT_EQ((node->type_string() == "Abs" || node->type_string() == "Softplus"), NodeIsMarkedForClustering(node));
  }

  NGraphClusterManager::EvictAllClusters();
  UnsetEnvVariable("NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS");
  RestoreEnv(env_map);
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