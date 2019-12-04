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

#include "ngraph_bridge/ngraph_backend_manager.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

Status GetPrefetchTensors(Graph* graph) {
  cout << "using prefetch env flag " << endl;
  // Set the prefetch shared obj if applicable
  NGraphPrefetchSharedResouce* shared_data = nullptr;
  Status s = ctx->resource_manager()->Lookup(
      NGraphPrefetchSharedResouce::CONTAINER_NAME,
      NGraphPrefetchSharedResouce::RESOURCE_NAME, &shared_data);

  if (!s.ok()) {
    // We are using this for the first time i.e., we need to do the following
    // 1. Create the shared data object
    // 2. We get another pipelined tensor pair for the current iteration and
    //   copy the TF tensor to this set and continue with the execution for
    //   for this iteration.
    shared_data = new NGraphPrefetchSharedResouce(
        name(), m_parallel_executor->GetOpBackendName(),
        m_parallel_executor->GetGraphId(),
        m_parallel_executor->GetNgraphClusterId());

    // Get the set of IO tensors for the next iteration
    std::tuple<int, PipelinedTensorVector, PipelinedTensorVector>
        io_tensors_next_iter;
    io_tensors_next_iter = pipelined_tensor_store->get_tensors();
    // Get prefetched inputs
    vector<shared_ptr<ng::runtime::Tensor>> pipelined_input_tensors_next_iter =
        get<1>(io_tensors_next_iter);
    vector<shared_ptr<ng::runtime::Tensor>> prefetched_input_tensors_next_iter =
        tensor_manager->GetPrefetchedTensors(pipelined_input_tensors_next_iter);

    // Save the prefetched input ngTensors for the next iteration
    NGraphPrefetchSharedResouce::InputTensorBundle next_input_tensor_bundle{
        get<0>(io_tensors_next_iter), prefetched_input_tensors_next_iter};

    OP_REQUIRES(ctx,
                current_iter_pipeline_depth == (!next_input_tensor_bundle.Id),
                errors::Internal("Current Pipeline Depth is ",
                                 current_iter_pipeline_depth,
                                 " and next iter pipeline depth is also  ",
                                 next_input_tensor_bundle.Id));

    shared_data->AddNextInputTensorBundleForDeviceTransfer(
        next_input_tensor_bundle);

    ctx->SetStatus(ctx->resource_manager()->Create(
        NGraphPrefetchSharedResouce::CONTAINER_NAME,
        NGraphPrefetchSharedResouce::RESOURCE_NAME, shared_data));
    // Continue the execution with the currently supplied TF tensor for the
    // last time
    NGRAPH_VLOG(2) << "[PREFETCH] COMPUTE: Creating the shared object to "
                      "signal prefetching";
  } else {
    cout << "using prefetch inputs " << endl;

    int prefetch_buffer_depth = shared_data->GetBufferDepth();
    int skip_count = shared_data->GetSkipCount();
    NGRAPH_VLOG(2) << "[PREFETCH] COMPUTE: DEPTH: " << prefetch_buffer_depth
                   << " skip count; " << skip_count;
    if (skip_count >= prefetch_buffer_depth) {
      cout << "skip_tf2ng_copy true " << endl;
      // We have been using the pipelined tensors - therefore do the
      // following:
      // 1. Save the prefetched Input tensors for the current iteration
      //    to the shared data object so that the prefetcher
      //    can continue with copying the next set of inout tensor to the
      //    device
      // 3. Execute the nGraph call for this iteration using the
      //    nG prefeteched input tensors we got from the shared data

      // Add the current prefetched tensors for the next iteration
      // Get prefetched inputs
      vector<shared_ptr<ng::runtime::Tensor>> prefetched_input_tensors =
          tensor_manager->GetPrefetchedTensors(ng_inputs);
      NGraphPrefetchSharedResouce::InputTensorBundle
          prefetch_input_tensor_bundle{current_iter_pipeline_depth,
                                       prefetched_input_tensors};
      shared_data->AddNextInputTensorBundleForDeviceTransfer(
          prefetch_input_tensor_bundle);

      // Update the input_tensors with the one ready for exdcution
      auto ng_input_tensor_bundle_ready =
          shared_data->GetNextInputTensorBundleReadyForDeviceExecution();
      current_iter_pipeline_depth = ng_input_tensor_bundle_ready.Id;
      vector<shared_ptr<ng::runtime::Tensor>> ng_prefetched_inputs =
          ng_input_tensor_bundle_ready.Inputs;
      OP_REQUIRES(ctx, current_iter_pipeline_depth ==
                           (!prefetch_input_tensor_bundle.Id),
                  errors::Internal("Current Pipeline Depth is ",
                                   current_iter_pipeline_depth,
                                   " and next iter pipeline depth is ", "also ",
                                   prefetch_input_tensor_bundle.Id));
      skip_tf2ng_copy = true;
      NGRAPH_VLOG(2) << "[PREFETCH] COMPUTE: Using device tensors";
    }
    shared_data->IncrSkipCount();
  }
}
}

}  // namespace ngraph_bridge
}  // namespace tensorflow
