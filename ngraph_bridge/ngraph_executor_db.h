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

#ifndef NGRAPH_EXECUTOR_DB_H_
#define NGRAPH_EXECUTOR_DB_H_
#pragma once

#include <mutex>
#include <ostream>
#include <vector>
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/graph.h"

#include "ngraph/ngraph.hpp"

#include "logging/ngraph_log.h"
#include "ngraph_bridge/ngraph_freshness_tracker.h"
#include "ngraph_bridge/ngraph_pipelined_tensors.h"

namespace tensorflow {

namespace ngraph_bridge {

class NGraphExecutorDB {
 public:
  ~NGraphExecutorDB() {
    m_executable_pipelined_tensors_map.clear();
    m_ng_function_map.clear();
    m_ng_exec_map.clear();
  }
  bool MaybeGetNgExecutable(std::string signature,
                            std::shared_ptr<ngraph::runtime::Executable>&
                                ng_exec)  // line no. 216, 220
  {
    lock_guard<mutex> lock(m_mutex);
    auto it = m_ng_exec_map.find(signature);
    if (it == m_ng_exec_map.end()) {
      return false;
    }
    ng_exec = it->second;
    UpdateLRU(signature);
    return true;
  }
  // make pair func and exece
  void AddItem(std::string signature,
               std::shared_ptr<ngraph::runtime::Executable> ng_exec,
               std::shared_ptr<ngraph::Function> ng_function,
               std::shared_ptr<ngraph::runtime::Executable>& evicted_ng_exec,
               int depth) {
    lock_guard<mutex> lock(m_mutex);
    const char* cache_depth_specified =
        std::getenv("NGRAPH_TF_FUNCTION_CACHE_ITEM_DEPTH");
    if (cache_depth_specified != nullptr) {
      if (m_ng_exec_map.size() >= atoi(cache_depth_specified)) {
        RemoveItem(m_lru.back(), evicted_ng_exec);
      }
    }
    m_ng_exec_map.emplace(signature, ng_exec);        // line no. 324
    m_ng_function_map.emplace(ng_exec, ng_function);  // line no. 327
    m_lru.push_front(signature);
    auto it =
        m_executable_pipelined_tensors_map.find(ng_exec);  // line no. 443-444

    // IsPipelinedTensorsStoreAvailable
    if (it == m_executable_pipelined_tensors_map.end()) {
      size_t num_inputs = ng_exec->get_parameters().size();
      size_t num_outputs = ng_exec->get_results().size();
      PipelinedTensorMatrix pipelined_input_tensors(num_inputs);
      PipelinedTensorMatrix pipelined_output_tensors(num_outputs);
      for (size_t i = 0; i < num_inputs; i++) {
        pipelined_input_tensors[i] = ng_exec->create_input_tensor(i, depth);
      }
      for (size_t i = 0; i < num_outputs; i++) {
        pipelined_output_tensors[i] = ng_exec->create_output_tensor(i, depth);
      }
      // InsertExecPipelineTesornMap
      shared_ptr<PipelinedTensorsStore> pts(new PipelinedTensorsStore(
          pipelined_input_tensors, pipelined_output_tensors));
      m_executable_pipelined_tensors_map.emplace(ng_exec, pts);
    }
  }

  bool MaybeGetNgFunction(
      std::shared_ptr<ngraph::runtime::Executable> ng_exec,
      std::shared_ptr<ngraph::Function>& ng_function)  // line no. 363, 364
  {
    lock_guard<mutex> lock(m_mutex);
    auto it = m_ng_function_map.find(ng_exec);
    if (it == m_ng_function_map.end()) return false;
    ng_function = it->second;
    return true;
  }

  bool GetDeviceTensors(
      const std::shared_ptr<ngraph::runtime::Executable>& ng_exec,
      std::tuple<int, PipelinedTensorVector, PipelinedTensorVector>&
          io_tensors) {
    PipelinedTensorsStore* pts(nullptr);
    try {
      lock_guard<mutex> lock(m_mutex);
      const auto& item = m_executable_pipelined_tensors_map.at(ng_exec);
      pts = item.get();
    } catch (...) {
      NGRAPH_VLOG(4) << "Error: "
                     << "Executable not found in the cache";
      return false;
    }
    io_tensors = pts->get_tensors();
    if (std::get<0>(io_tensors) < 0) {
      NGRAPH_VLOG(4) << "Error: "
                     << " Internal (No free tensor available) ";
      return false;
    }
    return true;
  }

 private:
  mutex m_mutex;
  std::list<std::string> m_lru;
  std::unordered_map<std::string, std::shared_ptr<ngraph::runtime::Executable>>
      m_ng_exec_map;
  std::unordered_map<std::shared_ptr<ngraph::runtime::Executable>,
                     std::shared_ptr<ngraph::Function>>
      m_ng_function_map;
  std::unordered_map<std::shared_ptr<ngraph::runtime::Executable>,
                     shared_ptr<PipelinedTensorsStore>>
      m_executable_pipelined_tensors_map;

  void RemoveItem(std::string signature,
                  std::shared_ptr<ngraph::runtime::Executable>&
                      evicted_ng_exec)  // line no. 261, 262, 263
  {
    // lock_guard<mutex> lock(m_mutex1);
    evicted_ng_exec = m_ng_exec_map[signature];
    m_ng_exec_map.erase(signature);
    m_ng_function_map.erase(evicted_ng_exec);
    m_lru.pop_back();
  }

  void UpdateLRU(std::string signature) {
    // lock_guard<mutex> lock(m_mutex);
    if (signature != m_lru.front()) {
      m_lru.remove(signature);
      m_lru.push_front(signature);
    }
  }
};

}  // namespace ngraph_bridge

}  // namespace tensorflow
#endif  // NGRAPH_EXECUTOR_H_
