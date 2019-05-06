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
#pragma once

#ifndef NGRAPH_TF_INPUT_DATA_PIPELINE_H_
#define NGRAPH_TF_INPUT_DATA_PIPELINE_H_

#include "ngraph/ngraph.hpp"
#include "tensorflow/core/graph/graph.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

class NGraphInputDataPiepline {
 public:
  // Should be used to load the input data on device prior to execution
  // 1. Creates ng-tensor of the specified backend
  // 2. Calls ng-tensor->write to copy the data (in the provided tf-tensor) from
  // host to device
  // 3. Adds the node name that produces this tensor and the ng-tensor to the
  // catalog
  static Status LoadInputDataOnDevice(vector<string>& input_node_names,
                                      vector<Tensor*>& input_tf_tensors,
                                      string backend_name = "");
};
}  // namespace ngraph_bridge

}  // namespace tensorflow
#endif  // NGRAPH_TF_INPUT_DATA_PIPELINE_H_
