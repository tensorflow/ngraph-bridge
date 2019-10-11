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
namespace testing {
class NGraphExecutorDBTest_CompileExe_Test;
class NGraphExecutorDBTest_CompileAndGetTensorsMultiThreaded_Test;
}
class NGraphExecutorDB {
 public:
  NGraphExecutorDB(int depth);
  ~NGraphExecutorDB();

  // Function returns true or false,
  // whether executable exists in map or not
  // And reference to ng_executable if it exists
  bool MaybeGetNgExecutable(
      std::string signature,
      std::shared_ptr<ngraph::runtime::Executable>& ng_exec);

  // Add items in m_ng_exec_map, m_ng_function_map,
  // and m_executable_pipelined_tensors_map
  void AddItem(std::string signature,
               std::pair<std::shared_ptr<ngraph::runtime::Executable>,
                         std::shared_ptr<ngraph::Function>>
                   ng_exec_func,
               std::shared_ptr<ngraph::runtime::Executable>& evicted_ng_exec);

  // Function returns true or false,
  // whether executable exists in map or not
  // And reference to ng_function if it exists
  bool MaybeGetNgFunction(std::shared_ptr<ngraph::runtime::Executable> ng_exec,
                          std::shared_ptr<ngraph::Function>& ng_function);

  // Returns Status and reference to io tensors
  Status GetDeviceTensors(
      const std::shared_ptr<ngraph::runtime::Executable>& ng_exec,
      std::tuple<int, PipelinedTensorVector, PipelinedTensorVector>&
          io_tensors);

 private:
  // For a gtest fixture TEST_F(NGraphExecutorDBTest, CompileExe),
  // the class name becomes:
  // tensorflow::ngraph_bridge::testing::NGraphExecutorDBTest_CompileExe_Test,
  // so this is the class that needs to be friended.
  friend class tensorflow::ngraph_bridge::testing::
      NGraphExecutorDBTest_CompileExe_Test;
  friend class tensorflow::ngraph_bridge::testing::
      NGraphExecutorDBTest_CompileAndGetTensorsMultiThreaded_Test;

  mutex m_mutex;
  int m_depth;
  // list contains most recently used signatures
  std::list<std::string> m_lru;

  // Map caontains signature and ng_executable
  std::unordered_map<std::string, std::shared_ptr<ngraph::runtime::Executable>>
      m_ng_exec_map;

  // Map conatains ng_executable and ng_function
  std::unordered_map<std::shared_ptr<ngraph::runtime::Executable>,
                     std::shared_ptr<ngraph::Function>>
      m_ng_function_map;

  // Map conations ng_executable and its corresponding pipeline tensors
  std::unordered_map<std::shared_ptr<ngraph::runtime::Executable>,
                     shared_ptr<PipelinedTensorsStore>>
      m_executable_pipelined_tensors_map;

  // Removes data from m_ng_exec_map,
  // m_ng_function_map,
  // m_executable_pipelined_tensors_map
  void RemoveItem(
      std::string signature,
      std::shared_ptr<ngraph::runtime::Executable>& evicted_ng_exec);

  // Pushes most recently used signature in m_lru list
  void UpdateLRU(std::string signature);
};

}  // namespace ngraph_bridge

}  // namespace tensorflow
#endif  // NGRAPH_EXECUTOR_H_
