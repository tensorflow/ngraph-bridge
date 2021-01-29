/*******************************************************************************
 * Copyright 2017-2020 Intel Corporation
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

#include <fstream>
#include <ostream>
#include <sstream>

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/tensor_coding.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"

#include "ngraph/ngraph.hpp"

using namespace std;

namespace tensorflow {
namespace ngraph_bridge {
namespace tf_utils {

bool IsAlreadyProcessed(Graph* g);

// Descending sort the map based on the value
void PrintNodeHistogram(const std::unordered_map<string, int>&,
                        bool sorted = true);

Status TensorToStream(std::ostream& ostream, const Tensor& tensor);

// Converts a TensorFlow DataType to an nGraph element::Type. Returns
// errors::Unimplemented if the element type is not supported by nGraph
// Core. Otherwise returns Status::OK().
Status TFDataTypeToNGraphElementType(DataType tf_dt,
                                     ngraph::element::Type* ng_et);

// Converts a TensorFlow TensorShape to an nGraph Shape. Requires that none of
// the dimension lengths in tf_shape are negative.
Status TFTensorShapeToNGraphShape(const TensorShape& tf_shape,
                                  ngraph::Shape* ng_shape);

// Dump TF graphs in .pbtxt format
void DumpTFGraph(tensorflow::Graph* graph, int idx, string filename_prefix);

}  // namespace tf_utils
}  // namespace ngraph_bridge
}  // namespace tensorflow
