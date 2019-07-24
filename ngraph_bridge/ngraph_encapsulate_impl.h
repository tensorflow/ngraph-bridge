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

#ifndef NGRAPH_TF_ENCAPSULATE_IMPL_H_
#define NGRAPH_TF_ENCAPSULATE_IMPL_H_

#include <ostream>
#include <vector>

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/graph.h"

#include "ngraph/ngraph.hpp"

#include "logging/ngraph_log.h"
#include "ngraph_bridge/ngraph_freshness_tracker.h"

namespace tensorflow {

namespace ngraph_bridge {

using NgFunctionIOCache = std::unordered_map<
    std::shared_ptr<ngraph::runtime::Executable>,
    std::vector<std::pair<void*, shared_ptr<ng::runtime::Tensor>>>>;

class NGraphEncapsulateImpl {
 public:
  // Ngraph Encapsulate Implementation class for EncapsulateOp class
  explicit NGraphEncapsulateImpl(string name);

  // Get tensorflow input tensors, input shapes, static_inputs to Compute
  // Signature
  Status ComputeSignature(const std::vector<Tensor>& tf_input_tensors,
                          std::vector<TensorShape>& input_shapes,
                          std::vector<const Tensor*>& static_input_map,
                          std::stringstream& signature_ss);

  // Calls Compute Signature and gets ngraph executable
  Status GetNgExecutable(const std::vector<Tensor>& tf_input_tensors,
                         std::vector<TensorShape>& input_shapes,
                         std::vector<const Tensor*>& static_input_map,
                         ng::runtime::Backend*& op_backend,
                         std::shared_ptr<ngraph::runtime::Executable>& ng_exec);

  // Allocate tensors for input arguments. Creates ngraph input tensors using
  // tensorflow tensors required to execute ngraph function
  Status AllocateNGInputTensors(
      const std::vector<Tensor>& tf_input_tensors,
      const std::shared_ptr<ngraph::runtime::Executable>& ng_exec,
      ng::runtime::Backend* op_backend,
      vector<shared_ptr<ng::runtime::Tensor>>& ng_inputs);

  // Allocate tensors for output results.  Creates ngraph output tensors using
  // tensorflow tensors required to execute ngraph function
  Status AllocateNGOutputTensors(
      const std::vector<Tensor*>& tf_output_tensors,
      const std::shared_ptr<ngraph::runtime::Executable>& ng_exec,
      ng::runtime::Backend* op_backend,
      vector<shared_ptr<ng::runtime::Tensor>>& ng_outputs);

  // Get current ngraph tensor
  std::shared_ptr<ng::runtime::Tensor> GetCurrentNgTensor(
      void* current_tf_ptr, void* last_tf_ptr,
      const std::shared_ptr<ng::runtime::Tensor>& last_ng_tensor,
      const bool& output_tensor,
      const std::shared_ptr<ngraph::runtime::Executable>& ng_exec,
      ng::runtime::Backend* op_backend,
      const ng::element::Type& ng_element_type, const ng::Shape& ng_shape);

  // TF Graph for the cluster
  Graph m_graph;
  // Freshness tracker maintains a set of ng::functions using a particular base
  // pointer(for Tensor)
  // A single instance of freshness_tracker is used across all
  // nGraphEncapsulateOp and nGraphVariable op
  NGraphFreshnessTracker* m_freshness_tracker;
  string m_name;
  std::mutex m_compute_lock;

  // Accessors(getters and setters) for the private data members of
  // NgraphEncapsulateImpl class
  // needed by
  // NgraphEncapsulateOp class
  const int& get_number_of_copies() { return number_of_copies; }

  void set_number_of_copies(const int& number) { number_of_copies = number; }

  const int& get_ngraph_cluster() { return m_ngraph_cluster; }

  void set_ngraph_cluster(const int& cluster) { m_ngraph_cluster = cluster; }

  const int& get_graph_id() { return m_graph_id; }

  void set_graph_id(const int& graph_id) { m_graph_id = graph_id; }

  const int& get_function_cache_depth_in_items() {
    return my_function_cache_depth_in_items;
  }

  const int& get_number_outputs() { return m_number_outputs; }

  const int& get_instance_id() { return my_instance_id; }

  const string& get_op_backend_name() { return m_op_backend_name; }

  void set_op_backend_name(const string& backend_name) {
    m_op_backend_name = backend_name;
  }

  bool get_log_copies() { return log_copies; }

  const std::vector<bool> get_static() { return m_input_is_static; }

  void resize_static(const int& size) { m_input_is_static.resize(size); }
  void set_static(const int& index, bool value) {
    m_input_is_static[index] = value;
  }

  std::unordered_map<std::string, std::shared_ptr<ngraph::runtime::Executable>>
  get_ng_exec_map() {
    return m_ng_exec_map;
  }

  void set_ng_exec_map(
      const std::string& ng_map_key,
      const std::shared_ptr<ngraph::runtime::Executable>& exec) {
    m_ng_exec_map[ng_map_key] = exec;
  }

  std::unordered_map<std::shared_ptr<ngraph::runtime::Executable>,
                     std::shared_ptr<ngraph::Function>>
  get_ng_function_map() {
    return m_ng_function_map;
  }

  void set_ng_function_map(
      const std::shared_ptr<ngraph::runtime::Executable>& exec,
      const std::shared_ptr<ngraph::Function>& function) {
    m_ng_function_map[exec] = function;
  }

  // TODO:sindhu have another get function for output_cache which is only
  // readable
  std::vector<std::pair<void*, shared_ptr<ng::runtime::Tensor>>>&
  get_ng_exec_output_cache_map(
      std::shared_ptr<ngraph::runtime::Executable> exec) {
    return m_ng_exec_output_cache_map[exec];
  }

  void set_ng_exec_output_cache_map(
      const std::shared_ptr<ngraph::runtime::Executable>& exec,
      const std::vector<std::pair<void*, shared_ptr<ng::runtime::Tensor>>>&
          cache) {
    m_ng_exec_output_cache_map[exec] = cache;
  }

 private:
  int number_of_copies = 0;
  int m_ngraph_cluster{-1};
  int m_graph_id{-1};
  int my_function_cache_depth_in_items = 16;
  int m_number_outputs = -1;
  int my_instance_id{0};
  string m_op_backend_name;
  std::stringstream copy_log_str;
  bool log_copies = false;
  std::vector<bool> m_input_is_static;
  std::list<std::string> m_lru;
  static int s_instance_count;

  // ng_function, ng_executable, Output and Input Cache maps
  std::unordered_map<std::string, std::shared_ptr<ngraph::runtime::Executable>>
      m_ng_exec_map;
  std::unordered_map<std::shared_ptr<ngraph::runtime::Executable>,
                     std::shared_ptr<ngraph::Function>>
      m_ng_function_map;

  NgFunctionIOCache m_ng_exec_input_cache_map;
  NgFunctionIOCache m_ng_exec_output_cache_map;
};

}  // namespace ngraph_bridge

}  // namespace tensorflow
#endif  // NGRAPH_TF_ENCAPSULATE_IMPL_H_
