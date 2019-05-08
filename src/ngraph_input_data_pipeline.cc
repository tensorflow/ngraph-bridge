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
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/graph/graph.h"

#include "ngraph/event_tracing.hpp"
#include "ngraph/ngraph.hpp"

#include "ngraph_backend_manager.h"
#include "ngraph_catalog.h"
#include "ngraph_input_data_pipeline.h"
#include "ngraph_utils.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

Status NGraphInputDataPiepline::LoadInputDataOnDevice(
    vector<string>& input_node_names, vector<Tensor*>& input_tf_tensors,
    string backend_name) {
  ngraph::Event load_data_event("LoadInputDataOnDevice", "", "");

  if (input_node_names.size() != input_tf_tensors.size()) {
    return errors::Internal(
        "Number of Input Node Names and Tensors don't match");
  };

  // if the backend is empty get currently set backend
  if (backend_name.empty()) {
    backend_name = BackendManager::GetCurrentlySetBackendName();
  }

  NGRAPH_VLOG(5) << "Got backend " << backend_name;
  BackendManager::CreateBackend(backend_name);
  ng::runtime::Backend* op_backend = BackendManager::GetBackend(backend_name);
  bool is_cpu_backend = backend_name == "CPU";

  // Where to release the backend?
  // Currently, released in the encapsulate op that uses these inputs

  // create ng-tensor and load to device
  for (int i = 0; i < input_tf_tensors.size(); i++) {
    // TF datatype to nGraph element type
    DataType tf_dtype = input_tf_tensors[i]->dtype();
    ng::element::Type ng_element_type;
    TF_RETURN_IF_ERROR(
        TFDataTypeToNGraphElementType(tf_dtype, &ng_element_type));

    // TF TensorShape to nGraphShape
    TensorShape tf_shape = input_tf_tensors[i]->shape();
    ng::Shape ng_shape(tf_shape.dims());
    for (int j = 0; j < tf_shape.dims(); ++j) {
      ng_shape[j] = tf_shape.dim_size(j);
    }

    shared_ptr<ng::runtime::Tensor> ng_tensor = nullptr;
    void* current_src_ptr = (void*)DMAHelper::base(input_tf_tensors[i]);
    if (is_cpu_backend) {
      // Create nGTensor
      ng_tensor =
          op_backend->create_tensor(ng_element_type, ng_shape, current_src_ptr);
    } else {
      // Create nGTensor
      ng_tensor = op_backend->create_tensor(ng_element_type, ng_shape);

      // Load to Device
      ng_tensor->write(current_src_ptr, 0,
                       ng_tensor->get_element_count() * ng_element_type.size());
    }

    // Save in Catalog
    NGraphCatalog::AddToInputDataTensorMap(input_node_names[i], ng_tensor);
  }

  load_data_event.Stop();
  ngraph::Event::write_trace(load_data_event);
  return Status::OK();
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
