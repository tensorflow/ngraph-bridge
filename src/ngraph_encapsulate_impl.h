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
#include "ngraph_freshness_tracker.h"
#include "ngraph_log.h"

namespace tensorflow {

using NgFunctionIOCache = std::unordered_map<
    std::shared_ptr<ngraph::runtime::Executable>,
    std::vector<std::pair<void*, shared_ptr<ng::runtime::Tensor>>>>;

namespace ngraph_bridge {

class NGraphEncapsulateImpl {
 public:
  explicit NGraphEncapsulateImpl(string name);

  Status ComputeSignature(std::vector<Tensor>& input_tensors,
                          std::vector<TensorShape>& input_shapes,
                          std::vector<const Tensor*>& static_input_map,
                          std::stringstream& signature_ss);

  Status GetNgExecutable(std::vector<Tensor>& input_tensors,
                         const std::pair<string, int64> ctx_params,
                         std::vector<TensorShape>& input_shapes,
                         std::vector<const Tensor*>& static_input_map,
                         ng::runtime::Backend*& op_backend,
                         std::shared_ptr<ngraph::runtime::Executable>& ng_exec);

  Status AllocateNGInputTensors(
      const std::vector<Tensor>& tf_input_tensors,
      std::shared_ptr<ngraph::runtime::Executable>& ng_exec,
      std::vector<TensorShape>& input_shapes, ng::runtime::Backend* op_backend,
      vector<shared_ptr<ng::runtime::Tensor>>& ng_inputs);

  Status AllocateNGOutputTensors(
      std::vector<Tensor*>& tf_output_tensors,
      std::vector<ng::element::Type> expected_output_types,
      const std::shared_ptr<ngraph::runtime::Executable>& ng_exec,
      std::vector<TensorShape>& input_shapes, ng::runtime::Backend* op_backend,
      vector<shared_ptr<ng::runtime::Tensor>>& ng_outputs,
      std::vector<std::pair<void*, std::shared_ptr<ng::runtime::Tensor>>>&
          output_caches);

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

  // Accessors(getters and setters) for the private data members needed by
  // NgraphEncapsulateOp class
  int get_number_of_copies() { return number_of_copies; }

  int set_number_of_copies(int number) {
    number_of_copies = number;
    return number_of_copies;
  }

  int get_ngraph_cluster() { return m_ngraph_cluster; }

  int set_ngraph_cluster(int cluster) {
    m_ngraph_cluster = cluster;
    return m_ngraph_cluster;
  }

  int get_graph_id() { return m_graph_id; }

  int set_graph_id(int graph_id) {
    m_graph_id = graph_id;
    return m_graph_id;
  }

  int get_function_cache_depth_in_items() {
    return my_function_cache_depth_in_items;
  }

  int get_number_outputs() { return m_number_outputs; }

  int get_instance_id() { return my_instance_id; }

  string get_op_backend_name() { return m_op_backend_name; }

  void set_op_backend_name(string backend_name) {
    m_op_backend_name = backend_name;
  }

  bool get_log_copies() { return log_copies; }

  std::vector<bool> get_static() { return m_input_is_static; }

  std::unordered_map<std::string, std::shared_ptr<ngraph::runtime::Executable>>
  get_ng_exec_map() {
    return m_ng_exec_map;
  }
  std::unordered_map<std::shared_ptr<ngraph::runtime::Executable>,
                     std::shared_ptr<ngraph::Function>>
  get_ng_function_map() {
    return m_ng_function_map;
  }
  NgFunctionIOCache get_ng_exec_output_cache_map() {
    return m_ng_exec_output_cache_map;
  }

 private:
  int number_of_copies = 0;
  int m_ngraph_cluster{-1};
  int m_graph_id{-1};
  int my_function_cache_depth_in_items = 16;
  int m_number_outputs = -1;
  string m_op_backend_name;
  std::stringstream copy_log_str;
  bool log_copies = false;
  std::vector<bool> m_input_is_static;
  std::list<std::string> m_lru;
  static int s_instance_count;
  int my_instance_id{0};

  // cache maps
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
