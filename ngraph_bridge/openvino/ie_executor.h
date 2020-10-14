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

#ifndef IE_EXECUTOR_H_
#define IE_EXECUTOR_H_

#include <ie_core.hpp>
#include <memory>
#include <string>
#include <vector>
#include "ngraph_bridge/openvino/ie_data.h"

class IE_Executor {
 public:
  IE_Executor(InferenceEngine::CNNNetwork ie_network, std::string device);
  ~IE_Executor();

  // Executes the inference
  void infer(std::vector<std::shared_ptr<IE_Data>> inputs,
             std::vector<std::shared_ptr<IE_Data>> outputs,
             std::vector<std::shared_ptr<IE_Data>> hoisted_params);

  // Returns output batch size based on the input batch size and the device
  // FIXME: This may not be needed
  size_t getOutputBatchSize(size_t inputBatchSize, std::string device) const;

 private:
  InferenceEngine::CNNNetwork m_network;
  std::vector<InferenceEngine::InferRequest> m_infer_reqs;
  std::string m_device;
  InferenceEngine::ExecutableNetwork m_exe_network;

  void start_async_inference(const int req_id);
  void complete_async_inference(const int req_id);
};

#endif  // IE_EXECUTOR_H_
