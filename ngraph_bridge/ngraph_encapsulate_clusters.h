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

#ifndef NGRAPH_TF_BRIDGE_ENCAPSULATE_CLUSTERS_H_
#define NGRAPH_TF_BRIDGE_ENCAPSULATE_CLUSTERS_H_
#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

#include <iostream>
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

namespace ngraph_bridge {

typedef std::map<std::string, std::vector<int>> ShapeHintMap;

// the integer represent AOT level requested.
typedef std::pair<bool, std::set<ShapeHintMap>> AOTInfo;

// TODO: an optimization would be to separate the analysis and rewriting passes
// cleanly, so that analysis pass is run in mark_for_clustering, and its
// information is reused here instead of recalculating

// TODO separate the AOT part out to a new function

/// Takes a TF graph where ngraph_cluster attributes has been marked in a
/// preceeding pass (assign_clusters), then replaces TF subgraphs and inserts
/// encapsulate ops in their place. Optionally can perform ahead of time
/// compilation. Optionally it can be used as an analysis pass to get the TF
/// subgraphs, but not perform the rewriting.
Status EncapsulateClusters(
    Graph* graph, int graph_id, FunctionDefLibrary* fdeflib,
    std::unordered_map<std::string, std::string> device_config,
    const AOTInfo& aot_info, bool analysis_pass);

Status PerformAOTOnEncapsulates(Graph* graph, const AOTInfo& aot_info);

}  // namespace ngraph_bridge
}  // namespace tensorflow

#endif  // NGRAPH_TF_BRIDGE_ENCAPSULATE_CLUSTERS_H_
