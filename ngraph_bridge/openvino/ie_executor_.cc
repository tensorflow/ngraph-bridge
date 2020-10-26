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

#include "ngraph_bridge/openvino/ie_executor.h"
#include <iostream>
#include "ngraph_bridge/openvino/ie_manager.h"

IE_Executor::IE_Executor(InferenceEngine::CNNNetwork ie_network,
                         std::string device)
    : m_network(ie_network), m_device(device) {
  if (std::getenv("NGRAPH_TF_DUMP_GRAPHS")) {
    auto& name = m_network.getName();
    m_network.serialize(name + ".xml", name + ".bin");
  }

  InferenceEngine::Core ie;
  // Load network to the plugin (m_device)
  m_exe_network = ie.LoadNetwork(m_network, m_device);
}

IE_Executor::~IE_Executor() {}

void IE_Executor::infer(std::vector<std::shared_ptr<IE_Data>> inputs,
                        std::vector<std::shared_ptr<IE_Data>> outputs,
                        std::vector<std::shared_ptr<IE_Data>> hoisted_params) {

  int num_req = 1;
  int batch_size = 0;

  while (m_infer_reqs.size() < num_req) {
    m_infer_reqs.push_back(m_exe_network.CreateInferRequest());
  }

  // Check if the number of inputs that the CNN network expects is equal to the
  // sum of the
  // inputs specified and the inputs we hoisted, if any.
  //InferenceEngine::InputsDataMap input_info = m_network.getInputsInfo();
  //if (input_info.size() != (inputs.size() + m_hoisted_params.size())) {
  //  THROW_IE_EXCEPTION
  //      << "Function inputs number differ from number of given inputs";
  //}
  //      std::cout << "Execute Inference - B" << std::endl;

  //  Prepare input blobs

  std::vector<InferenceEngine::MemoryBlob::Ptr> in_blobs(inputs.size()*num_req);
  std::vector<InferenceEngine::MemoryBlob::Ptr> param_blobs(
      hoisted_params.size());
  std::vector<InferenceEngine::MemoryBlob::Ptr> out_blobs(outputs.size()*num_req);
  std::cout << "Execute Inference - C" << std::endl;
  for (int i = 0; i < inputs.size(); i++) {
    InferenceEngine::SizeVector input_shape = inputs[i]->get_shape();
    InferenceEngine::Precision input_precision = inputs[i]->get_precision();
    InferenceEngine::Layout input_layout = inputs[i]->get_layout();
    const void* input_data_pointer = inputs[i]->get_data_pointer();
    size_t size = inputs[i]->get_byte_size();
    std::string input_name = inputs[i]->get_name();

    InferenceEngine::SizeVector req_shape(input_shape);
    if (batch_size != 0)
      req_shape[0] = batch_size;
    InferenceEngine::TensorDesc desc(input_precision, req_shape, input_layout);
    for (int j = 0; j < num_req; j++) {
      size_t req_size = size / num_req;
      const void* data_ptr =
          (void*)((uint64_t)(input_data_pointer) + req_size * j);
      int in_idx = i * num_req + j;
      IEManager::CreateBlob(desc, input_precision, data_ptr, req_size,
                            in_blobs[in_idx]);
      m_infer_reqs[j].SetBlob(input_name, in_blobs[in_idx]);
    }
  }
  for (int i = 0; i < hoisted_params.size(); i++) {
    InferenceEngine::SizeVector param_shape = hoisted_params[i]->get_shape();
    InferenceEngine::Precision param_precision = hoisted_params[i]->get_precision();
    InferenceEngine::Layout param_layout = hoisted_params[i]->get_layout();
    const void* param_data_pointer = hoisted_params[i]->get_data_pointer();
    size_t size = hoisted_params[i]->get_byte_size();
    std::string param_name = hoisted_params[i]->get_name();

    InferenceEngine::SizeVector req_shape(param_shape);
    InferenceEngine::TensorDesc desc(param_precision, req_shape, param_layout);
    IEManager::CreateBlob(desc, param_precision, param_data_pointer, size,
                          param_blobs[i]);
    for (int j = 0; j < num_req; j++) {
      m_infer_reqs[j].SetBlob(param_name, param_blobs[i]);
    }
  }
        std::cout << "Execute Inference - G" << std::endl;
  for (int i = 0; i < outputs.size(); i++) {
    InferenceEngine::SizeVector output_shape = outputs[i]->get_shape();
    InferenceEngine::Precision output_precision = outputs[i]->get_precision();
    InferenceEngine::Layout output_layout = outputs[i]->get_layout();
    const void* output_data_pointer = outputs[i]->get_data_pointer();
    size_t size = outputs[i]->get_byte_size();
    std::string output_name = outputs[i]->get_name();

    InferenceEngine::SizeVector req_shape(output_shape);
    if (batch_size != 0)
      req_shape[0] = batch_size;
    InferenceEngine::TensorDesc desc(output_precision, req_shape,
                                     output_layout);
    for (int j = 0; j < num_req; j++) {
      size_t req_size = size / num_req;
      const void* data_ptr =
          (void*)((uint64_t)(output_data_pointer) + req_size * j);
      int out_idx = i * num_req + j;
      IEManager::CreateBlob(desc, output_precision, data_ptr, req_size,
                            out_blobs[out_idx]);
      m_infer_reqs[j].SetBlob(output_name, out_blobs[out_idx]);
    }
  }
        std::cout << "Execute Inference - F" << std::endl;

  //m_infer_req.Infer();
  //m_ie_executor->infer(ie_inputs, ie_outputs, ie_hoisted_params);
  // Start Inference Requests
  for (int i = 0; i < num_req; i++) {
    start_async_inference(i);
  }
  // Complete Inference Requests
  for (int i = 0; i < num_req; i++) {
    complete_async_inference(i);
  }

  for (int i = 0; i < in_blobs.size(); i++) {
    in_blobs[i]->deallocate();
  }
  for (int i = 0; i < out_blobs.size(); i++) {
    out_blobs[i]->deallocate();
  }
  for (int i = 0; i < param_blobs.size(); i++) {
    param_blobs[i]->deallocate();
  }
        std::cout << "Execute Inference (check) - END" << std::endl;
}

//void IE_Executor::infer(std::vector<std::shared_ptr<IE_Data>> inputs,
//                        std::vector<std::shared_ptr<IE_Data>> outputs,
//                        std::vector<std::shared_ptr<IE_Data>> hoisted_params) {
//        std::cout << "Infer - A" << std::endl;
//  int batch_size = 0;
//  int num_req = 1;
//  //if (inputs.size() == 1 && inputs[0]->get_shape().size() > 1) {
//  //    batch_size =
//  //        IEManager::GetInputBatchSize(inputs[0]->get_shape()[0], m_device);
//  //          std::cout << "Infer - B" << std::endl;
//  //    num_req = inputs[0]->get_shape()[0] / batch_size;
//  //}
//  //      std::cout << "Infer - C" << std::endl;
//  //std::cout << "Running Inference - Device: " << m_device
//  //          << ", Num Req: " << num_req
//  //          //<< ", Batch Size: " << inputs[0]->get_shape()[0]
//  //          << ", Req Batch Size: " << batch_size << std::endl;
//
//  while (m_infer_reqs.size() < num_req) {
//    m_infer_reqs.push_back(m_exe_network.CreateInferRequest());
//  }
//        std::cout << "Infer - D" << std::endl;
//
//  InferenceEngine::InputsDataMap input_info = m_network.getInputsInfo();
//  if (input_info.size() != inputs.size()) {
//    THROW_IE_EXCEPTION
//        << "Function inputs number differ from number of given inputs";
//  }
//
//  std::vector<InferenceEngine::MemoryBlob::Ptr> in_blobs(num_req *
//                                                         inputs.size());
//  std::vector<InferenceEngine::MemoryBlob::Ptr> out_blobs(num_req *
//                                                          outputs.size());
//  std::vector<InferenceEngine::MemoryBlob::Ptr> param_blobs(
//      hoisted_params.size());
//
//  size_t i = 0;
//  for (const auto& it : input_info) {
//    InferenceEngine::SizeVector input_shape = inputs[i]->get_shape();
//    InferenceEngine::Precision input_precision = inputs[i]->get_precision();
//    InferenceEngine::Layout input_layout = inputs[i]->get_layout();
//    const void* input_data_pointer = inputs[i]->get_data_pointer();
//    size_t size = inputs[i]->get_byte_size();
//    std::string input_name =
//        inputs[i]->has_name() ? inputs[i]->get_name() : it.first;
//
//    InferenceEngine::SizeVector req_shape(input_shape);
//    if (batch_size != 0)
//      req_shape[0] = batch_size;
//    InferenceEngine::TensorDesc desc(input_precision, req_shape, input_layout);
//
//    for (int j = 0; j < num_req; j++) {
//      size_t req_size = size / num_req;
//      const void* data_ptr =
//          (void*)((uint64_t)(input_data_pointer) + req_size * j);
//      int in_idx = i * num_req + j;
//      IEManager::CreateBlob(desc, input_precision, data_ptr, req_size,
//                            in_blobs[in_idx]);
//      m_infer_reqs[j].SetBlob(input_name, in_blobs[in_idx]);
//    }
//    i++;
//  }
//
//  for (i = 0; i < hoisted_params.size(); i++) {
//    InferenceEngine::SizeVector param_shape = hoisted_params[i]->get_shape();
//    InferenceEngine::Precision param_precision =
//        hoisted_params[i]->get_precision();
//    InferenceEngine::Layout param_layout = hoisted_params[i]->get_layout();
//    const void* param_data_pointer = hoisted_params[i]->get_data_pointer();
//    size_t size = hoisted_params[i]->get_byte_size();
//    std::string param_name = hoisted_params[i]->get_name();
//
//    InferenceEngine::TensorDesc desc(param_precision, param_shape,
//                                     param_layout);
//    IEManager::CreateBlob(desc, param_precision, param_data_pointer, size,
//                          param_blobs[i]);
//    for (int j = 0; j < num_req; j++) {
//      m_infer_reqs[j].SetBlob(param_name, param_blobs[i]);
//    }
//  }
//
//  //  Prepare output blobs
//  InferenceEngine::OutputsDataMap output_info = m_network.getOutputsInfo();
//  if (output_info.size() != outputs.size()) {
//    THROW_IE_EXCEPTION
//        << "Function outputs number differ from number of given outputs";
//  }
//
//  i = 0;
//  for (const auto& it : output_info) {
//    InferenceEngine::SizeVector output_shape = outputs[i]->get_shape();
//    InferenceEngine::Precision output_precision = outputs[i]->get_precision();
//    InferenceEngine::Layout output_layout = outputs[i]->get_layout();
//    const void* output_data_pointer = outputs[i]->get_data_pointer();
//    size_t size = outputs[i]->get_byte_size();
//    std::string output_name =
//        outputs[i]->has_name() ? outputs[i]->get_name() : it.first;
//
//    InferenceEngine::SizeVector req_shape(output_shape);
//    if (batch_size != 0)
//      req_shape[0] = batch_size;
//    InferenceEngine::TensorDesc desc(output_precision, req_shape,
//                                     output_layout);
//
//    for (int j = 0; j < num_req; j++) {
//      size_t req_size = size / num_req;
//      const void* data_ptr =
//          (void*)((uint64_t)(output_data_pointer) + req_size * j);
//      int out_idx = i * num_req + j;
//      IEManager::CreateBlob(desc, output_precision, data_ptr, req_size,
//                            out_blobs[out_idx]);
//      m_infer_reqs[j].SetBlob(output_name, out_blobs[out_idx]);
//    }
//    i++;
//  }
//
//  // Start Inference Requests
//  for (i = 0; i < num_req; i++) {
//    start_async_inference(i);
//  }
//  // Complete Inference Requests
//  for (i = 0; i < num_req; i++) {
//    complete_async_inference(i);
//  }
//
//  for (i = 0; i < in_blobs.size(); i++) {
//    in_blobs[i]->deallocate();
//  }
//  for (i = 0; i < out_blobs.size(); i++) {
//    out_blobs[i]->deallocate();
//  }
//  for (i = 0; i < param_blobs.size(); i++) {
//    param_blobs[i]->deallocate();
//  }
//}

void IE_Executor::start_async_inference(const int req_id) {
  // Start Async inference
  try {
    m_infer_reqs[req_id].StartAsync();
    // std::cout << "Start Async Inference" << std::endl;
  } catch (InferenceEngine::details::InferenceEngineException e) {
    THROW_IE_EXCEPTION << "Couldn't start Inference: ";
  } catch (...) {
    THROW_IE_EXCEPTION << "Couldn't start Inference: ";
  }
}

void IE_Executor::complete_async_inference(const int req_id) {
  // Wait for Async inference completion
  try {
    // std::cout << "waiting for req " << req_id << std::endl;
    m_infer_reqs[req_id].Wait(
        InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
    // std::cout << "Complete Async Inference" << std::endl;
  } catch (InferenceEngine::details::InferenceEngineException e) {
    THROW_IE_EXCEPTION << " Exception with completing Inference: ";
  } catch (...) {
    THROW_IE_EXCEPTION << " Exception with completing Inference: ";
  }
}

size_t IE_Executor::getOutputBatchSize(size_t inputBatchSize,
                                       std::string device) const {
  return m_network.getBatchSize() *
         IEManager::GetNumRequests(inputBatchSize, device);
}
