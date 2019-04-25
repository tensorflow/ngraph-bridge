/*******************************************************************************
 * Copyright 2017-2018 Intel Corporation
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
#include "ngraph_cluster_manager.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {
// Static initializers
std::map<int, GraphDef*> NGraphClusterManager::s_cluster_graphs;
std::mutex NGraphClusterManager::s_cluster_graphs_mutex;

int NGraphClusterManager::NewCluster() {
  std::lock_guard<std::mutex> guard(s_cluster_graphs_mutex);

  int new_idx = s_cluster_graphs.size();
  s_cluster_graphs[new_idx] = new GraphDef();
  return new_idx;
}

GraphDef* NGraphClusterManager::GetClusterGraph(int idx) {
  std::lock_guard<std::mutex> guard(s_cluster_graphs_mutex);
  auto itr = s_cluster_graphs.find(idx);
  return (itr == s_cluster_graphs.end()) ? nullptr : itr->second;
}

void NGraphClusterManager::EvictCluster(int idx) {
  std::lock_guard<std::mutex> guard(s_cluster_graphs_mutex);
  s_cluster_graphs.erase(idx);
}

vector<int> NGraphClusterManager::GetClusterIndexes() {
  std::lock_guard<std::mutex> guard(s_cluster_graphs_mutex);
  std::vector<int> cluster_indexes;
  for (const auto& key_val : s_cluster_graphs)
    cluster_indexes.push_back(key_val.first);
  return cluster_indexes;
}

void NGraphClusterManager::EvictAllClusters() { s_cluster_graphs.clear(); }

}  // namespace ngraph_bridge

}  // namespace tensorflow
