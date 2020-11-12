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

#include "ngraph_bridge/ie_basic_engine.h"
#include "ngraph_bridge/ie_utils.h"
#include <iostream>


namespace tensorflow {
namespace ngraph_bridge {


IE_Basic_Engine::IE_Basic_Engine(InferenceEngine::CNNNetwork ie_network,
                                 std::string device)
    : IE_Backend_Engine(ie_network, device) {
}

IE_Basic_Engine::~IE_Basic_Engine() {}

void IE_Basic_Engine::infer(std::vector<std::shared_ptr<IETensor>>& inputs,
                        std::vector<std::shared_ptr<IETensor>>& outputs,
                        std::vector<std::string>& output_names,
                        std::vector<std::shared_ptr<IETensor>>& hoisted_params) {

  // Create request
  if (m_infer_reqs.empty()) {
    m_infer_reqs.push_back(m_exe_network.CreateInferRequest());
  }

  std::vector<InferenceEngine::MemoryBlob::Ptr> in_blobs(inputs.size());
  std::vector<InferenceEngine::MemoryBlob::Ptr> param_blobs(
      hoisted_params.size());
  std::vector<InferenceEngine::MemoryBlob::Ptr> out_blobs(outputs.size());
  //  Prepare input blobs
  for (int i = 0; i < inputs.size(); i++) {
    InferenceEngine::SizeVector input_shape = inputs[i]->get_dims();
    InferenceEngine::Precision input_precision = inputs[i]->get_precision();
    InferenceEngine::Layout input_layout = inputs[i]->get_layout();
    const void* input_data_pointer = inputs[i]->get_data_ptr();
    size_t size = inputs[i]->get_byte_size();
    std::string input_name = inputs[i]->get_name();

    InferenceEngine::TensorDesc desc(input_precision, input_shape, input_layout);
    IE_Utils::CreateBlob(desc, input_precision, input_data_pointer, size,
                          in_blobs[i]);
    m_infer_reqs[0].SetBlob(input_name, in_blobs[i]);
  }
  for (int i = 0; i < hoisted_params.size(); i++) {
    InferenceEngine::SizeVector param_shape = hoisted_params[i]->get_dims();
    InferenceEngine::Precision param_precision =
        hoisted_params[i]->get_precision();
    InferenceEngine::Layout param_layout = hoisted_params[i]->get_layout();
    const void* param_data_pointer = hoisted_params[i]->get_data_ptr();
    size_t size = hoisted_params[i]->get_byte_size();
    std::string param_name = hoisted_params[i]->get_name();

    InferenceEngine::TensorDesc desc(param_precision, param_shape, param_layout);
    IE_Utils::CreateBlob(desc, param_precision, param_data_pointer, size,
                          param_blobs[i]);
    m_infer_reqs[0].SetBlob(param_name, param_blobs[i]);
  }

  // Prepare output blobs
  for (int i = 0; i < outputs.size(); i++) {
    out_blobs[i] = nullptr;
    if (outputs[i] != nullptr) {
      InferenceEngine::SizeVector output_shape = outputs[i]->get_dims();
      InferenceEngine::Precision output_precision = outputs[i]->get_precision();
      InferenceEngine::Layout output_layout = outputs[i]->get_layout();
      const void* output_data_pointer = outputs[i]->get_data_ptr();
      size_t size = outputs[i]->get_byte_size();
      std::string output_name = outputs[i]->get_name();

      InferenceEngine::TensorDesc desc(output_precision, output_shape,
                                       output_layout);
      IE_Utils::CreateBlob(desc, output_precision, output_data_pointer, size,
                            out_blobs[i]);
      m_infer_reqs[0].SetBlob(output_name, out_blobs[i]);
    }
  }

  // Start Inference Request
  start_async_inference(0);
  // Complete Inference Request
  complete_async_inference(0);

  // Set dynamic output blobs
  for (int i = 0; i < outputs.size(); i++) {
    if (outputs[i] == nullptr) {
      auto blob = InferenceEngine::as<InferenceEngine::MemoryBlob>(
          m_infer_reqs[0].GetBlob(output_names[i]));
      auto lm = blob->rwmap();
      uint8_t* data_ptr = lm.as<uint8_t*>();
      InferenceEngine::TensorDesc desc = blob->getTensorDesc();
      InferenceEngine::SizeVector shape = desc.getDims();
      InferenceEngine::Precision precision = desc.getPrecision();
      InferenceEngine::Layout layout = desc.getLayout();
      size_t out_size = blob->byteSize();
      outputs[i] = std::make_shared<IETensor>((void*)data_ptr, precision, layout, shape, out_size, output_names[i]);
    }
  }

  for (int i = 0; i < in_blobs.size(); i++) {
    in_blobs[i]->deallocate();
  }
  for (int i = 0; i < out_blobs.size(); i++) {
    if (out_blobs[i] != nullptr) {
      out_blobs[i]->deallocate();
    }
  }
  for (int i = 0; i < param_blobs.size(); i++) {
    param_blobs[i]->deallocate();
  }
}

}
}
