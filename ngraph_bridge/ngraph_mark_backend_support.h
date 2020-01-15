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

#pragma once

#ifndef NGRAPH_TF_MARK_BACKEND_SUPPORT_H_
#define NGRAPH_TF_MARK_BACKEND_SUPPORT_H_

#include <string>

#include "ngraph/ngraph.hpp"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"

#include "ngraph_bridge/ngraph_mark_for_clustering.h"

namespace ng = ngraph;
using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

// Notes:
// We will attempt to run TranslateGraph (TG) and get the initial ng function
// from the TF subgraph.
// If TG is successful: then we call GetBackendSupportInfoForNgfunction, and get
// information which nodes are supportable by this specific backend.
// In the future GetBackendSupportInfoForNgfunction will be replaced by an
// ngraph API
// If TG fails, we will choose the more conservative approach and use the static
// map TFtoNgraphOpMap.
// Reasons TG could fail are: static inputs, unknown input shapes etc. note that
// with DynamicTranslateGraph, some of these failure modes should go away

// Given an ngraph backend, and an ng function, mark nodes as supported or
// unsupported by that backend
// TODO replace by appropriate ngcore API when available.
Status GetBackendSupportInfoForNgfunction(
    const ng::runtime::Backend* op_backend,
    const shared_ptr<ng::Function>& ng_function, std::map<std::string, bool>&);

Status GetBackendSupportInfoForTFSubgraph(const ng::runtime::Backend*,
                                          GraphDef*,
                                          std::map<std::string, bool>&,
                                          const std::set<ShapeHintMap>& hints);

Status IsSupportedByBackend(const Node* node,
                            const ng::runtime::Backend* op_backend,
                            bool& is_supported);
Status IsSupportedByBackend(
    const Node* node, const ng::runtime::Backend* op_backend,
    const std::map<std::string, std::set<std::shared_ptr<ngraph::Node>>>&
        TFtoNgraphOpMap,
    bool& is_supported);

}  // namespace ngraph_bridge
}  // namespace tensorflow

#endif  // NGRAPH_TF_MARK_BACKEND_SUPPORT_H_