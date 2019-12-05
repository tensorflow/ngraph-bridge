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

#ifndef NGRAPH_TF_BRIDGE_GET_PIPELINED_TENSORS_H
#define NGRAPH_TF_BRIDGE_GET_PIPELINED_TENSORS_H

#pragma once

#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

namespace ngraph_bridge {

Status GetPipelinedIOTensorsReadyForExecution(
    OpKernelContext* ctx, const std::vector<Tensor>& tf_input_tensors,
    const shared_ptr<PipelinedTensorsStore>& pipelined_tensor_store,
    const shared_ptr<NGraphTensorManager>& tensor_manager,
    std::tuple<int, PipelinedTensorVector, PipelinedTensorVector>&
        pipelined_io_tensors);

}  // namespace ngraph_bridge
}  // namespace tensorflow

#endif  // NGRAPH_TF_BRIDGE_GET_PIPELINED_TENSORS_H
