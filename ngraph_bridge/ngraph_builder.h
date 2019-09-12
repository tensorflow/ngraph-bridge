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
#ifndef NGRAPH_TF_BRIDGE_BUILDER_H_
#define NGRAPH_TF_BRIDGE_BUILDER_H_

#include <ostream>
#include <vector>

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/graph.h"

#include "ngraph/ngraph.hpp"

#include "logging/ngraph_log.h"

namespace tensorflow {

namespace ngraph_bridge {

class Builder {
 public:
  static Status TranslateGraph(
      const std::vector<TensorShape>& inputs,
      const std::vector<const Tensor*>& static_input_map, const Graph* tf_graph,
      std::shared_ptr<ngraph::Function>& ng_function);

  using OpMap = std::unordered_map<std::string,
                                   std::vector<std::shared_ptr<ngraph::Node>>>;

  template <typename T>
  static void MakePadding(const std::string& tf_padding_type,
                          const ngraph::Shape& ng_image_shape,
                          const ngraph::Shape& ng_kernel_shape,
                          const ngraph::Strides& ng_strides,
                          const ngraph::Shape& ng_dilations,
                          T& ng_padding_below, T& ng_padding_above) {
    ngraph::Shape ng_dilation_kernel_shape{
        (ng_kernel_shape[0] - 1) * ng_dilations[0] + 1,
        (ng_kernel_shape[1] - 1) * ng_dilations[1] + 1};

    MakePadding(tf_padding_type, ng_image_shape, ng_dilation_kernel_shape,
                ng_strides, ng_padding_below, ng_padding_above);
  }

  template <typename T>
  static void MakePadding3D(const std::string& tf_padding_type,
                            const ngraph::Shape& ng_image_shape,
                            const ngraph::Shape& ng_kernel_shape,
                            const ngraph::Strides& ng_strides,
                            const ngraph::Shape& ng_dilations,
                            T& ng_padding_below, T& ng_padding_above) {
    ngraph::Shape ng_dilation_kernel_shape{
        (ng_kernel_shape[0] - 1) * ng_dilations[0] + 1,
        (ng_kernel_shape[1] - 1) * ng_dilations[1] + 1,
        (ng_kernel_shape[2] - 1) * ng_dilations[2] + 1};

    MakePadding3D(tf_padding_type, ng_image_shape, ng_dilation_kernel_shape,
                  ng_strides, ng_padding_below, ng_padding_above);
  }

  template <typename T>
  static void MakePadding(const std::string& tf_padding_type,
                          const ngraph::Shape& ng_image_shape,
                          const ngraph::Shape& ng_kernel_shape,
                          const ngraph::Strides& ng_strides,
                          T& ng_padding_below, T& ng_padding_above) {
    if (tf_padding_type == "SAME") {
      for (size_t i = 0; i < 2; i++) {
        size_t image_size = ng_image_shape[i];
        size_t filter_shape = ng_kernel_shape[i];
        size_t filter_stride = ng_strides[i];

        int64 padding_needed;
        if (image_size % filter_stride == 0) {
          padding_needed = filter_shape - filter_stride;
        } else {
          padding_needed = filter_shape - (image_size % filter_stride);
        }
        if (padding_needed < 0) {
          padding_needed = 0;
        }

        size_t padding_lhs = padding_needed / 2;
        size_t padding_rhs = padding_needed - padding_lhs;
        ng_padding_below[i] = padding_lhs;
        ng_padding_above[i] = padding_rhs;
      }
    }

    NGRAPH_VLOG(3) << "ng_padding_below: " << ngraph::join(ng_padding_below);
    NGRAPH_VLOG(3) << "ng_padding_above: " << ngraph::join(ng_padding_above);
  }

  template <typename T>
  static void MakePadding3D(const std::string& tf_padding_type,
                            const ngraph::Shape& ng_image_shape,
                            const ngraph::Shape& ng_kernel_shape,
                            const ngraph::Strides& ng_strides,
                            T& ng_padding_below, T& ng_padding_above) {
    if (tf_padding_type == "SAME") {
      for (size_t i = 0; i < 3; i++) {
        size_t image_size = ng_image_shape[i];
        size_t filter_shape = ng_kernel_shape[i];
        size_t filter_stride = ng_strides[i];

        int64 padding_needed;
        if (image_size % filter_stride == 0) {
          padding_needed = filter_shape - filter_stride;
        } else {
          padding_needed = filter_shape - (image_size % filter_stride);
        }
        if (padding_needed < 0) {
          padding_needed = 0;
        }

        size_t padding_lhs = padding_needed / 2;
        size_t padding_rhs = padding_needed - padding_lhs;
        ng_padding_below[i] = padding_lhs;
        ng_padding_above[i] = padding_rhs;
      }
    }

    NGRAPH_VLOG(3) << "ng_padding_below: " << ngraph::join(ng_padding_below);
    NGRAPH_VLOG(3) << "ng_padding_above: " << ngraph::join(ng_padding_above);
  }

  static const std::map<
      DataType,
      std::pair<std::function<Status(const Node*, ngraph::element::Type,
                                     std::shared_ptr<ngraph::Node>*)>,
                const ngraph::element::Type>>&
  TF_NGRAPH_CONST_MAP();

  template <class TOpType, class... TArg>
  static std::shared_ptr<TOpType> ConstructNgNode(const std::string& op_name,
                                                  TArg&&... Args) {
    auto ng_node = std::make_shared<TOpType>(std::forward<TArg>(Args)...);
    ng_node->set_friendly_name(op_name);
    ng_node->add_provenance_tag(op_name);
    InsertInLogs(op_name, ng_node->get_name());
    return ng_node;
  }

  static const std::map<string, std::vector<string>>&
  GetTFNodeToNgNodeConversionTable() {
    return tf_node_to_ng_node;
  }

 private:
  static void ComputeScaleOffsetFolded(const uint& num_bits,
                                       const bool& unsigned_type,
                                       const bool& scaled, const int min_range,
                                       const int max_range, float* scale,
                                       int* offset);
  static std::map<string, std::vector<string>> tf_node_to_ng_node;
  static void InsertInLogs(string tf_op_name, string ng_op_name) {
    auto itr = tf_node_to_ng_node.find(tf_op_name);
    if (itr == tf_node_to_ng_node.end()) {
      tf_node_to_ng_node.insert({tf_op_name, {ng_op_name}});
    } else {
      itr->second.push_back(ng_op_name);
    }
  }
};

}  // namespace ngraph_bridge

}  // namespace tensorflow

#endif
