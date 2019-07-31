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

#ifndef NGRAPH_TF_BRIDGE_ENCAPSULATE_CLUSTERS_H_
#define NGRAPH_TF_BRIDGE_ENCAPSULATE_CLUSTERS_H_
#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

namespace ngraph_bridge {

// TODO unit test the class and move to a different file
class PartialShape {
  // This is a simple class that can represent full or partial shapes
  // a full shape has all dimensions >= 0. A partial shape has atleast one
  // dimension < 0
  // Its built like an optional/maybe, so please use is_valid before accessing
  // other functions

 public:
  PartialShape(std::vector<int> shape, bool valid = true)
      : m_shape(shape), m_valid(valid) {}
  PartialShape() : m_valid(false) {}

  PartialShape(tensorflow::TensorShapeProto tensor_shape_proto) {
    try {
      m_shape.resize(tensor_shape_proto.dim_size());
      for (uint shape_idx = 0; shape_idx < tensor_shape_proto.dim_size();
           shape_idx++) {
        auto num_elems_in_this_dim = tensor_shape_proto.dim(shape_idx).size();
        m_shape.push_back(num_elems_in_this_dim);
        // -1 means not specified
      }
      m_valid = true;
    } catch (...) {
      invalidate();
    }
  }

  bool is_concrete() const {
    check_valid();
    return std::all_of(m_shape.begin(), m_shape.end(),
                       [](int i) { return i >= 0; });
  };

  int size() const {
    check_valid();
    return m_shape.size();
  }

  int operator[](int idx) const {
    check_valid();
    return m_shape[idx];
  }

  std::vector<int> get_shape_vector() const {
    check_valid();
    return m_shape;
  }

  bool is_valid() const { return m_valid; }

  void concretize(PartialShape shape_hint) {
    check_valid();
    uint base_rank = m_shape.size();
    if (base_rank != shape_hint.size()) {  // different ranks
      invalidate();
      return;
    } else {
      for (int i = 0; i < base_rank; i++) {
        if (m_shape[i] != shape_hint[i]) {
          if (m_shape[i] == -1 && shape_hint[i] > -1) {
            m_shape[i] = shape_hint[i];
          } else {
            invalidate();
            return;
          }
        }
      }
      return;
    }
  }

 private:
  std::vector<int> m_shape;
  bool m_valid;

  void check_valid() const {
    if (!m_valid) {
      throw std::runtime_error(
          string("Attempted to use an invalid PartialShape"));
    }
  }

  void invalidate() {
    m_shape.clear();
    m_valid = false;
  }
};

typedef std::map<std::string, std::vector<int>> ShapeHintMap;
typedef std::pair<bool, std::set<ShapeHintMap>> AOTInfo;

Status EncapsulateClusters(
    Graph* graph, int graph_id, FunctionDefLibrary* fdeflib,
    std::unordered_map<std::string, std::string> device_config,
    AOTInfo aot_info);

}  // namespace ngraph_bridge
}  // namespace tensorflow

#endif  // NGRAPH_TF_BRIDGE_ENCAPSULATE_CLUSTERS_H_
