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

#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb_text.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/edgeset.h"
#include "tensorflow/core/lib/core/errors.h"

#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/builder/numpy_transpose.hpp"
#include "ngraph/builder/quantization.hpp"
#include "ngraph/op/argmax.hpp"
#include "ngraph/op/argmin.hpp"
#include "ngraph/op/util/logical_reduction.hpp"

#include "logging/ngraph_log.h"
#include "ngraph_bridge/ngraph_api.h"
#include "ngraph_bridge/ngraph_backend_manager.h"
#include "ngraph_bridge/ngraph_builder.h"
#include "ngraph_bridge/ngraph_conversions.h"
#include "ngraph_bridge/ngraph_mark_for_clustering.h"
#include "ngraph_bridge/ngraph_utils.h"

#if defined(NGRAPH_DISTRIBUTED)
#include "ngraph/distributed.hpp"
#endif

using tensorflow::int32;
using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

static bool VecStrCmp(const std::vector<string>& a,
                      const std::vector<string>& b) {
  return a == b;
}

static Status ValidateInputCount(const Node* op, tensorflow::int32 count) {
  if (op->num_inputs() != count) {
    return errors::InvalidArgument("\"", op->name(), "\" requires ", count,
                                   " input(s), got ", op->num_inputs(),
                                   " instead");
  }
  return Status::OK();
}

static Status ValidateInputCountMin(const Node* op, tensorflow::int32 count) {
  if (op->num_inputs() < count) {
    return errors::InvalidArgument("\"", op->name(), "\" requires at least ",
                                   count, " input(s), got ", op->num_inputs(),
                                   " instead");
  }
  return Status::OK();
}
//
// Helper for storing ops in ng_op_map.
// For most of the cases, op would have one output so
// vector ng_op_map[op_name] would contain one element.
//
// If storing more than one output_nodes, make sure it's in
// the same order as tensorflow would do that.
//
// Parameters:
//    Builder::OpMap& ng_op_map        - The TF-to-nGraph op map.
//    std::string op_name              - Name of the op.
//
//    shared_ptr<ng::Node> output_node - ng::Node to store
//

static void SaveNgOp(Builder::OpMap& ng_op_map, const std::string& op_name,
                     const shared_ptr<ng::Node>& output_node) {
  // no need to try-catch, map[key] will create vector object
  // if not exists
  ng_op_map[op_name].push_back(output_node);
}

void Builder::SetTracingInfo(const std::string& op_name,
                             const shared_ptr<ng::Node> ng_node) {
  ng_node->set_friendly_name(op_name);
  ng_node->add_provenance_tag(op_name);
  if (config::IsLoggingPlacement()) {
    cout << "TF_to_NG: " << op_name << " --> " << ng_node->get_name() << "\n";
  }
}

template <class TOpType, class... TArg>
std::shared_ptr<TOpType> ConstructNgNode(const std::string& op_name,
                                         TArg&&... Args) {
  auto ng_node = std::make_shared<TOpType>(std::forward<TArg>(Args)...);
  Builder::SetTracingInfo(op_name, ng_node);
  return ng_node;
}

// Helper for fetching correct input node from ng_op_map.
// Handles edge checking to make sure correct input node is
// fetched.
//
// Reduces some boilerplate code (incorrect from now) like this:
//
//      Node* tf_input;
//      TF_RETURN_IF_ERROR(op->input_node(0, &tf_input));
//
//      shared_ptr<ng::Node> ng_input;
//      try {
//        ng_input = ng_op_map.at(tf_input->name());
//      } catch (const std::out_of_range&) {
//        return errors::NotFound(tf_input->name(),
//                                    " is not found in the ng_op_map");
//      }
//
// Into 2 lines:
//
//      shared_ptr<ng::node> ng_input;
//      TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, &ng_input))
//
//
//
// Parameters:
//    Builder::OpMap& ng_op_map     - The TF-to-nGraph op map.
//    Node* op                  - TF op being translated.
//    input_idx                     - index of input
//
//    shared_ptr<ng::Node> *result  - ng::Node pointer where result
//                                    will be written
//
//

static Status GetInputNode(const Builder::OpMap& ng_op_map, const Node* op,
                           size_t input_idx, shared_ptr<ng::Node>* result) {
  // input op may have resulted in more than one ng::Node (eg. Split)
  // we need to look at Edge to check index of the input op
  std::vector<const Edge*> edges;
  TF_RETURN_IF_ERROR(op->input_edges(&edges));
  size_t src_output_idx;
  try {
    src_output_idx = edges.at(input_idx)->src_output();
  } catch (const out_of_range&) {
    return Status(error::NOT_FOUND, "Edge not found");
  }

  Node* tf_input;
  TF_RETURN_IF_ERROR(op->input_node(input_idx, &tf_input));
  const std::vector<shared_ptr<ng::Node>>* ng_op = nullptr;
  try {
    ng_op = &ng_op_map.at(tf_input->name());
  } catch (const out_of_range&) {
    return Status(error::NOT_FOUND,
                  string("Ngraph op not found for ") + tf_input->name());
  }
  try {
    *result = ng_op->at(src_output_idx);
  } catch (const out_of_range&) {
    return Status(error::NOT_FOUND, string("Input node not found at index ") +
                                        to_string(src_output_idx));
  }
  return Status::OK();
}

namespace detail {
static Status GetInputNodes(const Builder::OpMap&, const Node*, size_t) {
  return Status::OK();
}

template <typename... Arguments>
static Status GetInputNodes(const Builder::OpMap& ng_op_map, const Node* op,
                            size_t index, shared_ptr<ng::Node>* result,
                            Arguments&&... remaining) {
  if (result != nullptr) {
    TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, index, result));
  }
  return GetInputNodes(ng_op_map, op, index + 1, remaining...);
}
}  // namespace detail

template <typename... Arguments>
static Status GetInputNodes(const Builder::OpMap& ng_op_map, const Node* op,
                            Arguments&&... remaining) {
  constexpr size_t args_len = sizeof...(Arguments);
  TF_RETURN_IF_ERROR(ValidateInputCount(op, args_len));
  return detail::GetInputNodes(ng_op_map, op, 0, remaining...);
}

static Status GetStaticNodeTensor(
    const Node* node, const std::vector<const Tensor*>& static_input_map,
    Tensor* result) {
  if (node->type_string() == "_Arg") {
    int arg_index;
    TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "index", &arg_index));
    const Tensor* source_tensor = static_input_map[arg_index];
    if (source_tensor == nullptr) {
      return errors::Internal(
          "GetStaticNodeTensor called on _Arg but input tensor is missing from "
          "static input map");
    }
    *result = *source_tensor;
    return Status::OK();
  } else if (node->type_string() == "Const") {
    if (!result->FromProto(node->def().attr().at("value").tensor())) {
      return errors::Internal(
          "GetStaticNodeTensor: Const tensor proto parsing failed");
    }
    return Status::OK();
  } else {
    return errors::Internal("GetStaticNodeTensor called on node with type ",
                            node->type_string(), "; _Arg or Const expected");
  }
}

template <typename Ttensor, typename Tvector>
static void ConvertTensorDataToVector(const Tensor& tensor,
                                      std::vector<Tvector>* vector) {
  const Ttensor* data = tensor.flat<Ttensor>().data();
  vector->resize(tensor.NumElements());
  for (int64 i = 0; i < tensor.NumElements(); i++) {
    (*vector)[i] = Tvector(data[i]);
  }
}

template <typename T>
static Status TensorDataToVector(const Tensor& tensor, std::vector<T>* vector) {
  DataType dt = tensor.dtype();

  // If dt and T match, we can just copy.
  if (dt == DataTypeToEnum<T>::value) {
    *vector = std::vector<T>(tensor.flat<T>().data(),
                             tensor.flat<T>().data() + tensor.NumElements());
  }
  // Else we have to convert.
  else {
    switch (dt) {
      case DT_FLOAT:
        ConvertTensorDataToVector<float, T>(tensor, vector);
        break;
      case DT_DOUBLE:
        ConvertTensorDataToVector<double, T>(tensor, vector);
        break;
      case DT_INT8:
        ConvertTensorDataToVector<int8, T>(tensor, vector);
        break;
      case DT_INT16:
        ConvertTensorDataToVector<int16, T>(tensor, vector);
        break;
      case DT_INT32:
        ConvertTensorDataToVector<int32, T>(tensor, vector);
        break;
      case DT_INT64:
        ConvertTensorDataToVector<int64, T>(tensor, vector);
        break;
      case DT_UINT8:
        ConvertTensorDataToVector<uint8, T>(tensor, vector);
        break;
      case DT_UINT16:
        ConvertTensorDataToVector<uint16, T>(tensor, vector);
        break;
      case DT_UINT32:
        ConvertTensorDataToVector<uint32, T>(tensor, vector);
        break;
      case DT_UINT64:
        ConvertTensorDataToVector<uint64, T>(tensor, vector);
        break;
      case DT_BOOL:
        ConvertTensorDataToVector<bool, T>(tensor, vector);
        break;
      default:
        return errors::Internal("TensorDataToVector: tensor has element type ",
                                DataType_Name(dt), ", vector has type ",
                                DataType_Name(DataTypeToEnum<T>::value),
                                "; don't know how to convert");
    }
  }
  return Status::OK();
}

template <typename T>
static Status GetStaticInputVector(
    const Node* op, int64 input_index,
    const std::vector<const Tensor*>& static_input_map,
    std::vector<T>* vector) {
  Node* input_node;
  TF_RETURN_IF_ERROR(op->input_node(input_index, &input_node));
  Tensor input_tensor;
  TF_RETURN_IF_ERROR(
      GetStaticNodeTensor(input_node, static_input_map, &input_tensor));
  TF_RETURN_IF_ERROR(TensorDataToVector(input_tensor, vector));
  return Status::OK();
}

// Helper for Builder::TranslateGraph ("Const" op)
template <typename T, typename VecT = T>
static Status MakeConstOp(const Node* op, ng::element::Type et,
                          std::shared_ptr<ng::Node>* ng_node) {
  vector<VecT> const_values;
  TensorShapeProto shape_proto;

  TF_RETURN_IF_ERROR(
      ValuesFromConstNode<T, VecT>(op->def(), &shape_proto, &const_values));

  TensorShape const_shape(shape_proto);

  ng::Shape ng_shape;
  TF_RETURN_IF_ERROR(TFTensorShapeToNGraphShape(const_shape, &ng_shape));

  *ng_node =
      ConstructNgNode<ng::op::Constant>(op->name(), et, ng_shape, const_values);
  return Status::OK();
}

const std::map<DataType,
               std::pair<std::function<Status(const Node*, ng::element::Type,
                                              std::shared_ptr<ng::Node>*)>,
                         const ngraph::element::Type>>&
Builder::TF_NGRAPH_CONST_MAP() {
  static const std::map<
      DataType, std::pair<std::function<Status(const Node*, ng::element::Type,
                                               std::shared_ptr<ng::Node>*)>,
                          const ngraph::element::Type>>
      the_map = {
          {DataType::DT_FLOAT, make_pair(MakeConstOp<float>, ng::element::f32)},
          {DataType::DT_DOUBLE,
           make_pair(MakeConstOp<double>, ng::element::f64)},
          {DataType::DT_INT8, make_pair(MakeConstOp<int8>, ng::element::i8)},
          {DataType::DT_INT16, make_pair(MakeConstOp<int16>, ng::element::i16)},
          {DataType::DT_QINT8, make_pair(MakeConstOp<qint8>, ng::element::i8)},
          {DataType::DT_QUINT16,
           make_pair(MakeConstOp<quint8>, ng::element::u8)},
          {DataType::DT_INT32, make_pair(MakeConstOp<int32>, ng::element::i32)},
          {DataType::DT_INT64, make_pair(MakeConstOp<int64>, ng::element::i64)},
          {DataType::DT_UINT8, make_pair(MakeConstOp<uint8>, ng::element::u8)},
          {DataType::DT_UINT16,
           make_pair(MakeConstOp<uint16>, ng::element::u16)},
          {DataType::DT_BOOL,
           make_pair(MakeConstOp<bool, char>, ng::element::boolean)}};
  return the_map;
}

std::pair<std::shared_ptr<ng::Node>, std::shared_ptr<ng::Node>>
Builder::PerformNgBroadcast(const string& prov_tag,
                            std::shared_ptr<ng::Node> ng_lhs,
                            std::shared_ptr<ng::Node> ng_rhs) {
  // builder::numpy_broadcast is the only known builder that has the possibility
  // that the output node is same as the input node
  // So we take special care to check, before calling SetTracingInfo
  std::shared_ptr<ng::Node> ng_lhs_new, ng_rhs_new;
  std::tie(ng_lhs_new, ng_rhs_new) =
      ng::builder::numpy_broadcast(std::make_pair(ng_lhs, ng_rhs));
  if (ng_lhs_new != ng_lhs) {
    Builder::SetTracingInfo(prov_tag, ng_lhs_new);
  }
  if (ng_rhs_new != ng_rhs) {
    Builder::SetTracingInfo(prov_tag, ng_rhs_new);
  }
  return make_pair(ng_lhs_new, ng_rhs_new);
}

// Helper function to translate a unary op.
//
// Parameters:
//
//    Node* op                   - TF op being translated. Must have one input.
//    const std::vector<const Tensor*>& static_input_map
//                               - the static input map
//    Builder::OpMap& ng_op_map  - The TF-to-nGraph op map.
//
//    std::function<std::shared_ptr<ng::Node>(std::shared_ptr<ng::Node>>
//      create_unary_op           - Function to construct the graph implementing
//                                 the unary op, given the input to the unop
//                                 as an argument.
//
// Example usage:
//
//  if (n->type_string == "Square") {
//    TF_RETURN_IF_ERROR(TranslateUnaryOp(n, static_input_map, ng_op_map,
//                       [] (std::shared_ptr<ng::Node> n) {
//                           return (std::make_shared<ng::op::Multiply>(n,n));
//                       });
//  }
static Status TranslateUnaryOp(
    const Node* op, const std::vector<const Tensor*>&,
    Builder::OpMap& ng_op_map,
    std::function<std::shared_ptr<ng::Node>(std::shared_ptr<ng::Node>)>
        create_unary_op) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));
  auto ng_node = create_unary_op(ng_input);
  if (ng_node != ng_input) {
    Builder::SetTracingInfo(op->name(), ng_node);
  }
  SaveNgOp(ng_op_map, op->name(), ng_node);
  return Status::OK();
}

// Helper function to translate a unary op in cases where there is a one-to-one
// mapping from TensorFlow ops to nGraph ops.
//
// Example usage:
//
//  if (n->type_string == "Abs") {
//    TF_RETURN_IF_ERROR(TranslateUnaryOp<ng::op::Abs>(n, static_input_map,
//    ng_op_map));
//  }
//
template <typename T>
static Status TranslateUnaryOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateUnaryOp(op, static_input_map, ng_op_map,
                          [&op](std::shared_ptr<ng::Node> n) {
                            return ConstructNgNode<T>(op->name(), n);
                          });
}

// Helper function to translate a binary op
// Parameters:
//
//    Node* op               - TF op being translated. Must have only two
//    inputs.
//    const std::vector<const Tensor*>& static_input_map - the static input map
//    Builder::OpMap& ng_op_map  - The TF-to-nGraph op map.
//    std::function<std::shared_ptr<ng::Node>(std::shared_ptr<ng::Node>,
//    std::shared_ptr<ng::Node>)>
//    create_binary_op           - Function to construct the graph implementing
//                                 the binary op, given the 2 ng_inputs to the
//                                 binaryop
// Example Usage:
//
// if (op->type_string() == "SquaredDifference") {
//      TF_RETURN_IF_ERROR(TranslateBinaryOp(op, ng_op_map,
//         [](std::shared_ptr<ng::Node> ng_input1, std::shared_ptr<ng::Node>
//         ng_input2) {
//           auto ng_diff = std::make_shared<ng::op::Subtract>(input1, input2);
//           return std::make_shared<ng::op::Multiply>(ng_diff,ng_diff);
//         }));
//    }
//

static Status TranslateBinaryOp(
    const Node* op, const std::vector<const Tensor*>&,
    Builder::OpMap& ng_op_map,
    std::function<std::shared_ptr<ng::Node>(std::shared_ptr<ng::Node>,
                                            std::shared_ptr<ng::Node>)>
        create_binary_op) {
  std::shared_ptr<ng::Node> ng_lhs, ng_rhs;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_lhs, &ng_rhs));

  std::tie(ng_lhs, ng_rhs) =
      Builder::PerformNgBroadcast(op->name(), ng_lhs, ng_rhs);

  auto ng_node = create_binary_op(ng_lhs, ng_rhs);
  if (ng_node != ng_lhs && ng_node != ng_rhs) {
    Builder::SetTracingInfo(op->name(), ng_node);
  }

  SaveNgOp(ng_op_map, op->name(), ng_node);

  return Status::OK();
}

// Helper function to translate a binary op in cases where there is a one-to-one
// mapping from TensorFlow ops to nGraph ops.
//
// Example usage:
//
//  if (n->type_string == "Add") {
//    TF_RETURN_IF_ERROR(TranslateBinaryOp<ng::op::Add>(op, static_input_map,
//    ng_op_map));
//  }
//
template <typename T>
static Status TranslateBinaryOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateBinaryOp(
      op, static_input_map, ng_op_map, [&op](std::shared_ptr<ng::Node> ng_lhs,
                                             std::shared_ptr<ng::Node> ng_rhs) {
        return ConstructNgNode<T>(op->name(), ng_lhs, ng_rhs);
      });
}

// Helper function for translating QuantizedAvgPool and QuantizedMaxPool
static Status TranslateQuantizedPoolOp(const Node* op,
                                       const std::vector<const Tensor*>&,
                                       Builder::OpMap& ng_op_map,
                                       std::string pooling_name) {
  bool is_quantizedAvgPool = pooling_name == "QuantizedAvgPool";
  bool is_quantizedMaxPool = pooling_name == "QuantizedMaxPool";

  if (!(is_quantizedAvgPool || is_quantizedMaxPool)) {
    return errors::InvalidArgument(
        "Expected quantized pooling type node to be ScaledQuantizedAvgPool or "
        "ScaledQuantizedMaxPool");
  }
  shared_ptr<ng::Node> ng_input, ng_min, ng_max;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_min, &ng_max));
  std::vector<int32> tf_strides;
  std::vector<int32> tf_ksize;
  std::string tf_padding_type;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "ksize", &tf_ksize));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));

  NGRAPH_VLOG(3) << ng::join(tf_strides);
  NGRAPH_VLOG(3) << ng::join(tf_ksize);
  NGRAPH_VLOG(3) << tf_padding_type;

  bool is_nhwc = true;  // The input data format is always NHWC
  ng::Strides ng_strides(2);
  ng::Shape ng_image_shape(2);
  ng::Shape ng_kernel_shape(2);
  BatchedOpParamToNGraph(is_nhwc, tf_strides, ng_strides);
  BatchedOpParamToNGraph(is_nhwc, ng_input->get_output_shape(0),
                         ng_image_shape);
  BatchedOpParamToNGraph(is_nhwc, tf_ksize, ng_kernel_shape);
  BatchToNGraph(op->name(), is_nhwc, ng_input);

  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);
  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  ng::Shape ng_padding_below{0, 0};
  ng::Shape ng_padding_above{0, 0};
  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_padding_below, ng_padding_above);

  // Creating and passing dummy nodes to quantized pool operation because it
  // does
  // not use them. If it ever starts using min/max, the dummy min-max would
  // cause it to fail
  shared_ptr<ng::Node> dummy_min(nullptr), dummy_max(nullptr);

  std::shared_ptr<ng::Node> ng_quant_pool;
  if (is_quantizedAvgPool) {
    // QuantizeAvgPool
    // TF doesn't include padding in avg calculation
    ng_quant_pool = ng::builder::ScaledQuantizedAvgPool(
        ng_input, ng_kernel_shape, ng_strides, ng_padding_below,
        ng_padding_above, false, dummy_min, dummy_max);
  } else {
    // QuantizeMaxPool
    ng_quant_pool = ng::builder::ScaledQuantizedMaxPool(
        ng_input, ng_kernel_shape, ng_strides, ng_padding_below,
        ng_padding_above, dummy_min, dummy_max);
  }
  Builder::SetTracingInfo(op->name(), ng_quant_pool);

  BatchToTensorflow(op->name(), is_nhwc, ng_quant_pool);
  SaveNgOp(ng_op_map, op->name(), ng_quant_pool);
  // For QuantizedAvgPool and QuantizedMaxPool input min-max remains unchanged
  // and is just propagated along
  // https://github.com/tensorflow/tensorflow/blob/9590c4c32dd4346ea5c35673336f5912c6072bf2/tensorflow/core/kernels/quantized_pooling_ops.cc#L99
  SaveNgOp(ng_op_map, op->name(), ng_min);
  SaveNgOp(ng_op_map, op->name(), ng_max);
  return Status::OK();
}

static Status TranslateAddNOp(const Node* op, const std::vector<const Tensor*>&,
                              Builder::OpMap& ng_op_map) {
  std::vector<shared_ptr<ng::Node>> ng_arg_vec(op->num_inputs());

  for (int inp_idx = 0; inp_idx < op->num_inputs(); inp_idx++)
    TF_RETURN_IF_ERROR(
        GetInputNode(ng_op_map, op, inp_idx, &ng_arg_vec[inp_idx]));

  SaveNgOp(
      ng_op_map, op->name(),
      std::accumulate(std::next(ng_arg_vec.begin()), ng_arg_vec.end(),
                      ng_arg_vec.at(0),
                      [&op](shared_ptr<ng::Node> a, shared_ptr<ng::Node> b) {
                        return ConstructNgNode<ng::op::Add>(op->name(), a, b);
                      }));  // accumulation: start with
                            // first element. default op is
                            // addition
  return Status::OK();
}

template <typename T>
static Status TranslateArgMinMaxOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  bool is_argmin = std::is_same<T, ng::op::ArgMin>::value;
  bool is_argmax = std::is_same<T, ng::op::ArgMax>::value;
  if (!(is_argmin || is_argmax)) {
    return errors::InvalidArgument("Expected node to be argmin or argmax type");
  }

  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, &ng_input));

  std::vector<int64> tf_dim;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &tf_dim));

  ng::Shape input_shape = ng_input->get_shape();
  size_t input_rank = input_shape.size();

  if (tf_dim.size() != 1) {
    return errors::InvalidArgument(
        (is_argmin ? "ArgMin" : "ArgMax"),
        " Op: dimension must be scalar, operates on a single axis");
  }

  // If input dimension is negative, make it positive
  if (tf_dim[0] < 0) {
    tf_dim[0] = (int64)input_rank + tf_dim[0];
  }
  size_t input_dims = tf_dim[0];

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "output_type", &dtype));

  ng::element::Type ng_et;
  TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(dtype, &ng_et));

  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<T>(op->name(), ng_input, input_dims, ng_et));
  return Status::OK();
}

static Status TranslateAvgPoolOp(const Node* op,
                                 const std::vector<const Tensor*>&,
                                 Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  std::vector<int32> tf_strides;
  std::vector<int32> tf_ksize;
  std::string tf_padding_type;
  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "ksize", &tf_ksize));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument(
        "AvgPool data format is neither NHWC nor NCHW");
  }

  bool is_nhwc = (tf_data_format == "NHWC");

  NGRAPH_VLOG(3) << ng::join(tf_strides);
  NGRAPH_VLOG(3) << ng::join(tf_ksize);
  NGRAPH_VLOG(3) << tf_padding_type;
  NGRAPH_VLOG(3) << tf_data_format;

  ng::Strides ng_strides(2);
  ng::Shape ng_image_shape(2);
  ng::Shape ng_kernel_shape(2);
  BatchedOpParamToNGraph(is_nhwc, tf_strides, ng_strides);
  BatchedOpParamToNGraph(is_nhwc, ng_input->get_shape(), ng_image_shape);
  BatchedOpParamToNGraph(is_nhwc, tf_ksize, ng_kernel_shape);
  BatchToNGraph(op->name(), is_nhwc, ng_input);
  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);
  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  // TODO: change this once nGraph supports negative padding
  // (CoordinateDiff) for AvgPool
  // ng::CoordinateDiff ng_padding_below{0,0};
  // ng::CoordinateDiff ng_padding_above{0,0};
  ng::Shape ng_padding_below{0, 0};
  ng::Shape ng_padding_above{0, 0};

  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_padding_below, ng_padding_above);

  std::shared_ptr<ng::Node> ng_avgpool = ConstructNgNode<ng::op::AvgPool>(
      op->name(), ng_input, ng_kernel_shape, ng_strides, ng_padding_below,
      ng_padding_above, false);

  BatchToTensorflow(op->name(), is_nhwc, ng_avgpool);
  NGRAPH_VLOG(3) << "avgpool outshape: {" << ng::join(ng_avgpool->get_shape())
                 << "}";

  SaveNgOp(ng_op_map, op->name(), ng_avgpool);
  return Status::OK();
}

static Status TranslateAvgPoolGradOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_grad;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, nullptr, &ng_grad));

  std::vector<int32> tf_orig_input_shape_vec;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(op, 0, static_input_map, &tf_orig_input_shape_vec));

  std::vector<int32> tf_strides;
  std::vector<int32> tf_ksize;
  std::string tf_padding_type;
  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "ksize", &tf_ksize));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument(
        "AvgPoolGrad data format is neither NHWC nor NCHW");
  }

  bool is_nhwc = (tf_data_format == "NHWC");

  NGRAPH_VLOG(3) << ng::join(tf_strides);
  NGRAPH_VLOG(3) << ng::join(tf_ksize);
  NGRAPH_VLOG(3) << tf_padding_type;
  NGRAPH_VLOG(3) << tf_data_format;

  ng::Shape ng_orig_input_shape;
  for (size_t i = 0; i < tf_orig_input_shape_vec.size(); i++) {
    ng_orig_input_shape.push_back(tf_orig_input_shape_vec[i]);
  }

  ng::Shape ng_forward_arg_shape(4);
  ng::Strides ng_strides(2);
  ng::Shape ng_image_shape(2);
  ng::Shape ng_window_shape(2);

  BatchedOpParamReshape(is_nhwc, ng_orig_input_shape, ng_forward_arg_shape);
  BatchToNGraph(op->name(), is_nhwc, ng_grad);
  BatchedOpParamToNGraph(is_nhwc, tf_strides, ng_strides);
  BatchedOpParamToNGraph(is_nhwc, ng_orig_input_shape, ng_image_shape);
  BatchedOpParamToNGraph(is_nhwc, tf_ksize, ng_window_shape);

  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);
  NGRAPH_VLOG(3) << "ng_window_shape: " << ng::join(ng_window_shape);
  NGRAPH_VLOG(3) << "ng_forward_arg_shape: " << ng::join(ng_forward_arg_shape);

  // TODO: change this once nGraph supports negative padding
  // (CoordinateDiff) for AvgPool
  // ng::CoordinateDiff ng_padding_below{0,0};
  // ng::CoordinateDiff ng_padding_above{0,0};
  ng::Shape ng_padding_below{0, 0};
  ng::Shape ng_padding_above{0, 0};

  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_window_shape,
                       ng_strides, ng_padding_below, ng_padding_above);

  std::shared_ptr<ng::Node> ng_avgpool_backprop =
      ConstructNgNode<ng::op::AvgPoolBackprop>(
          op->name(), ng_forward_arg_shape, ng_grad, ng_window_shape,
          ng_strides, ng_padding_below, ng_padding_above, false);

  BatchToTensorflow(op->name(), is_nhwc, ng_avgpool_backprop);

  NGRAPH_VLOG(3) << "avgpoolbackprop outshape: {"
                 << ng::join(ng_avgpool_backprop->get_shape()) << "}";

  SaveNgOp(ng_op_map, op->name(), ng_avgpool_backprop);

  return Status::OK();
}

static Status TranslateBatchMatMulOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_lhs, ng_rhs;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_lhs, &ng_rhs));

  std::string backend_name;
  TF_RETURN_IF_ERROR(ngraph_bridge::GetNodeBackend(op, &backend_name));

  auto ng_lhs_shape = ng_lhs->get_shape();
  auto ng_rhs_shape = ng_rhs->get_shape();

  if (ng_lhs_shape.size() != ng_rhs_shape.size()) {
    return errors::InvalidArgument(
        "Dimensions of two input args are not the same for BatchMatMul");
  }
  size_t n_dims = ng_lhs_shape.size();
  if (n_dims < 2) {
    return errors::InvalidArgument(
        "Dimensions of input args for BatchMatMul must be >=2", n_dims);
  }

  ng::AxisVector out_axes;
  for (size_t i = 0; i < n_dims - 2; ++i) {
    if (ng_lhs_shape[i] != ng_rhs_shape[i]) {
      return errors::InvalidArgument(
          "ng_lhs_shape and ng_rhs_shape must be the same for BatchMatMul "
          "for each dimension",
          i);
    }
    out_axes.push_back(i);
  }

  bool tf_adj_x = false;
  bool tf_adj_y = false;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "adj_x", &tf_adj_x));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "adj_y", &tf_adj_y));

  auto ng_lhs_axes = out_axes;
  auto ng_rhs_axes = out_axes;

  // Get the backend name, if the backend is CPU and n_dims >= 3
  // then use the BatchMatMul op supported by nGraph
  if (n_dims >= 3 && backend_name == "CPU") {
    // Transpose X if AdjX = true
    if (tf_adj_x) {
      ng_lhs_axes.push_back(n_dims - 1);
      ng_lhs_axes.push_back(n_dims - 2);
      ng_lhs = ng::builder::numpy_transpose(ng_lhs, ng_lhs_axes);
      Builder::SetTracingInfo(op->name(), ng_lhs);
      ng_lhs_shape = ng_lhs->get_shape();
    } else {
      ng_lhs_axes.push_back(n_dims - 2);
      ng_lhs_axes.push_back(n_dims - 1);
    }
    // Transpose Y if AdjY = true
    if (tf_adj_y) {
      ng_rhs_axes.push_back(n_dims - 1);
      ng_rhs_axes.push_back(n_dims - 2);
      ng_rhs = ng::builder::numpy_transpose(ng_rhs, ng_rhs_axes);
      Builder::SetTracingInfo(op->name(), ng_rhs);
      ng_rhs_shape = ng_rhs->get_shape();
    } else {
      ng_rhs_axes.push_back(n_dims - 2);
      ng_rhs_axes.push_back(n_dims - 1);
    }

    if (n_dims == 3) {
      SaveNgOp(ng_op_map, op->name(), ConstructNgNode<ngraph::op::BatchMatMul>(
                                          op->name(), ng_lhs, ng_rhs));
    } else {
      // Find the compound size for dim1 so as to reshape to 3D
      size_t compound_size = 1;
      for (size_t i = 0; i < out_axes.size(); i++) {
        compound_size *= ng_lhs_shape[i];
      }

      ng::Shape tmp_lhs_shape = {compound_size, ng_lhs_shape[n_dims - 2],
                                 ng_lhs_shape[n_dims - 1]};
      ng::Shape tmp_rhs_shape = {compound_size, ng_rhs_shape[n_dims - 2],
                                 ng_rhs_shape[n_dims - 1]};

      auto output_shape = ng_lhs_shape;
      output_shape[n_dims - 1] = ng_rhs_shape[n_dims - 1];
      ng::AxisVector tmp_axes = {0, 1, 2};

      std::shared_ptr<ng::Node> lhs_reshape =
          ConstructNgNode<ngraph::op::Reshape>(op->name(), ng_lhs, ng_lhs_axes,
                                               tmp_lhs_shape);
      std::shared_ptr<ng::Node> rhs_reshape =
          ConstructNgNode<ngraph::op::Reshape>(op->name(), ng_rhs, ng_rhs_axes,
                                               tmp_rhs_shape);
      std::shared_ptr<ng::Node> batchmatmul =
          ConstructNgNode<ngraph::op::BatchMatMul>(op->name(), lhs_reshape,
                                                   rhs_reshape);
      SaveNgOp(ng_op_map, op->name(),
               ConstructNgNode<ngraph::op::Reshape>(op->name(), batchmatmul,
                                                    tmp_axes, output_shape));
    }
  } else {
    if (tf_adj_x) {
      ng_lhs_axes.push_back(n_dims - 1);
      ng_lhs_axes.push_back(n_dims - 2);
      ng_lhs = ng::builder::numpy_transpose(ng_lhs, ng_lhs_axes);
      Builder::SetTracingInfo(op->name(), ng_lhs);
    }
    if (tf_adj_y) {
      ng_rhs_axes.insert(ng_rhs_axes.begin(), n_dims - 2);
      ng_rhs_axes.insert(ng_rhs_axes.begin(), n_dims - 1);
      ng_rhs = ng::builder::numpy_transpose(ng_rhs, ng_rhs_axes);
      Builder::SetTracingInfo(op->name(), ng_rhs);
    } else {
      ng_rhs_axes.insert(ng_rhs_axes.begin(), n_dims - 1);
      ng_rhs_axes.insert(ng_rhs_axes.begin(), n_dims - 2);
      ng_rhs = ng::builder::numpy_transpose(ng_rhs, ng_rhs_axes);
      Builder::SetTracingInfo(op->name(), ng_rhs);
    }

    ng_lhs_shape = ng_lhs->get_shape();
    ng_rhs_shape = ng_rhs->get_shape();

    if (ng_lhs_shape[n_dims - 1] != ng_rhs_shape[0]) {
      return errors::InvalidArgument(
          "The last dimension of ng_lhs and the first dimension of ng_rhs "
          "should have the same size");
    }

    if (n_dims == 2) {
      SaveNgOp(ng_op_map, op->name(),
               ConstructNgNode<ngraph::op::Dot>(op->name(), ng_lhs, ng_rhs));
    } else {
      auto output_shape = ng_lhs_shape;
      output_shape[n_dims - 1] = ng_rhs_shape[1];
      auto dot_output =
          ConstructNgNode<ngraph::op::Dot>(op->name(), ng_lhs, ng_rhs);

      size_t compound_size = 1;
      for (size_t i = 0; i < out_axes.size(); i++) {
        compound_size *= output_shape[i];
      }
      auto dot_axes = out_axes;
      dot_axes.push_back(n_dims - 2);
      dot_axes.push_back(n_dims - 1);
      for (size_t i = 0; i < out_axes.size(); i++) {
        dot_axes.push_back(n_dims + i);
      }
      ng::Shape dot_shape = {compound_size, ng_lhs_shape[n_dims - 2],
                             ng_rhs_shape[1], compound_size};
      std::shared_ptr<ng::Node> dot_reshape;
      if (n_dims == 3) {
        dot_reshape = dot_output;
      } else {
        dot_reshape = ConstructNgNode<ngraph::op::Reshape>(
            op->name(), dot_output, dot_axes, dot_shape);
      }
      ng::Shape tmp_shape = {1, ng_lhs_shape[n_dims - 2], ng_rhs_shape[1]};
      vector<shared_ptr<ngraph::Node>> tmp_tensors;
      for (size_t i = 0; i < dot_shape[0]; i++) {
        const std::vector<size_t> lower_bound{i, 0, 0, i};
        const std::vector<size_t> upper_bound{i + 1, dot_shape[1], dot_shape[2],
                                              i + 1};
        auto slice_out = ConstructNgNode<ngraph::op::Slice>(
            op->name(), dot_reshape, lower_bound, upper_bound);
        auto reshape_out = ConstructNgNode<ngraph::op::Reshape>(
            op->name(), slice_out, ng::AxisVector{0, 1, 2, 3}, tmp_shape);
        tmp_tensors.push_back(reshape_out);
      }
      auto concat_op =
          ConstructNgNode<ngraph::op::Concat>(op->name(), tmp_tensors, 0);
      if (n_dims == 3) {
        SaveNgOp(ng_op_map, op->name(), concat_op);
      } else {
        SaveNgOp(
            ng_op_map, op->name(),
            ConstructNgNode<ngraph::op::Reshape>(
                op->name(), concat_op, ng::AxisVector{0, 1, 2}, output_shape));
      }
    }
  }
  return Status::OK();
}

static Status TranslateBiasAddOp(const Node* op,
                                 const std::vector<const Tensor*>&,
                                 Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_bias;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_bias));

  std::string tf_data_format;
  if (GetNodeAttr(op->attrs(), "data_format", &tf_data_format) !=
      Status::OK()) {
    tf_data_format = "NHWC";
  }

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument(
        "BiasAdd data format is neither NHWC nor NCHW");
  }

  auto ng_input_shape = ng_input->get_shape();
  auto ng_bias_shape = ng_bias->get_shape();
  if (ng_bias_shape.size() != 1) {
    return errors::InvalidArgument(
        "Bias argument to BiasAdd does not have one dimension");
  }

  bool is_nhwc = (tf_data_format == "NHWC");

  ng::AxisSet ng_broadcast_axes;

  if (is_nhwc) {
    for (size_t i = 0; i < ng_input_shape.size() - 1; i++) {
      ng_broadcast_axes.insert(i);
    }
  } else {
    for (size_t i = 0; i < ng_input_shape.size(); i++) {
      if (i != 1) {
        ng_broadcast_axes.insert(i);
      }
    }
  }

  auto ng_bias_broadcasted = ConstructNgNode<ng::op::Broadcast>(
      op->name(), ng_bias, ng_input_shape, ng_broadcast_axes);
  auto ng_add =
      ConstructNgNode<ng::op::Add>(op->name(), ng_input, ng_bias_broadcasted);

  SaveNgOp(ng_op_map, op->name(), ng_add);
  return Status::OK();
}

static Status TranslateBiasAddGradOp(const Node* op,
                                     const std::vector<const Tensor*>&,
                                     Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  std::string tf_data_format;
  if (GetNodeAttr(op->attrs(), "data_format", &tf_data_format) !=
      Status::OK()) {
    tf_data_format = "NHWC";
  }

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument(
        "BiasAddGrad data format is neither NHWC nor NCHW");
  }

  bool is_nhwc = (tf_data_format == "NHWC");

  ng::AxisSet reduction_axes;
  shared_ptr<ng::Node> ng_biasadd_backprop;
  auto ng_input_shape = ng_input->get_shape();

  if (is_nhwc) {
    if (ng_input_shape.size() < 2) {
      return errors::InvalidArgument(
          "BiasAddGrad argument needs to have at least 2 dimensions for NHWC "
          "data format");
    }
    for (size_t i = 0; i < ng_input_shape.size() - 1; i++) {
      reduction_axes.insert(i);
    }
  } else {
    // Tensorflow NCHW format supports only 4D input/output tensor
    if (ng_input_shape.size() != 4) {
      return errors::InvalidArgument(
          "BiasAddGrad only support 4d input/output for NCHW data format");
    }
    for (size_t i = 0; i < ng_input_shape.size(); i++) {
      if (i != ng_input_shape.size() - 3) reduction_axes.insert(i);
    }
  }

  ng_biasadd_backprop =
      ConstructNgNode<ng::op::Sum>(op->name(), ng_input, reduction_axes);

  SaveNgOp(ng_op_map, op->name(), ng_biasadd_backprop);
  return Status::OK();
}

static Status TranslateCastOp(const Node* op, const std::vector<const Tensor*>&,
                              Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "DstT", &dtype));

  ng::element::Type ng_et;
  TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(dtype, &ng_et));

  try {
    SaveNgOp(ng_op_map, op->name(),
             ConstructNgNode<ng::op::Convert>(op->name(), ng_input, ng_et));
  } catch (const std::out_of_range&) {
    return errors::Unimplemented("Unsupported TensorFlow data type: ",
                                 DataType_Name(dtype));
  }
  return Status::OK();
}
static Status TranslateCombinedNonMaxSuppressionOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_boxes, ng_scores;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_boxes, &ng_scores,
                                   nullptr, nullptr, nullptr, nullptr));

  std::vector<int> max_output_size_per_class;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 2, static_input_map,
                                          &max_output_size_per_class));
  std::vector<int> max_total_size;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(op, 3, static_input_map, &max_total_size));
  std::vector<float> iou_threshold;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(op, 4, static_input_map, &iou_threshold));

  std::vector<float> score_threshold;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(op, 5, static_input_map, &score_threshold));

  bool pad_per_class;
  if (GetNodeAttr(op->attrs(), "pad_per_class", &pad_per_class) !=
      Status::OK()) {
    pad_per_class = false;
  }
  bool clip_boxes;
  if (GetNodeAttr(op->attrs(), "clip_boxes", &clip_boxes) != Status::OK()) {
    clip_boxes = false;
  }
  // max_output_size_per_class must be scalar
  if (max_output_size_per_class.size() != 1) {
    return errors::InvalidArgument(
        "CombinedNonMaxSuppression Op: max_output_size_per_class of cnms must "
        "be scalar ",
        max_output_size_per_class.size());
  }
  // max_total_size must be scalar
  if (max_total_size.size() != 1) {
    return errors::InvalidArgument(
        "CombinedNonMaxSuppression Op: max_total_size of cnms must be scalar ",
        max_total_size.size());
  }
  // iou_threshold must be scalar
  if (iou_threshold.size() != 1) {
    return errors::InvalidArgument(
        "CombinedNonMaxSuppression Op: iou_threshold of cnms must be scalar ",
        iou_threshold.size());
  }

  // score_threshold must be scalar
  if (score_threshold.size() != 1) {
    return errors::InvalidArgument(
        "CombinedNonMaxSuppression Op: score_threshold of cnms must be scalar ",
        score_threshold.size());
  }

  std::string backend_name;
  TF_RETURN_IF_ERROR(ngraph_bridge::GetNodeBackend(op, &backend_name));

  auto config_map = BackendManager::GetBackendAttributeValues(backend_name);
  if (config_map.at("ngraph_backend") != "NNPI") {
    return errors::Internal("In translating CombinedNonMaxSuppression op ",
                            op->name(), " found requested backend ",
                            backend_name, " which is unsupported");
  }

  ng::runtime::Backend* backend = BackendManager::GetBackend(backend_name);

  shared_ptr<ng::Node> ng_cnms = backend->get_backend_op(
      "CombinedNonMaxSuppression", &ng_boxes, &ng_scores,
      (size_t)(max_output_size_per_class[0]), (size_t)(max_total_size[0]),
      (float)(iou_threshold[0]), (float)score_threshold[0], (bool)pad_per_class,
      (bool)clip_boxes);
  if (ng_cnms == nullptr) {
    return errors::Internal("In translating CombinedNonMaxSuppression op ",
                            op->name(),
                            " backend could not return valid ngraph node");
  }
  Builder::SetTracingInfo(op->name(), ng_cnms);
  shared_ptr<ngraph::Node> ng_nmsed_boxes =
      ConstructNgNode<ngraph::op::GetOutputElement>(op->name(), ng_cnms, 0);
  shared_ptr<ngraph::Node> ng_nmsed_scores =
      ConstructNgNode<ngraph::op::GetOutputElement>(op->name(), ng_cnms, 1);
  shared_ptr<ngraph::Node> ng_nmsed_classes =
      ConstructNgNode<ngraph::op::GetOutputElement>(op->name(), ng_cnms, 2);
  shared_ptr<ngraph::Node> ng_valid_detections =
      ConstructNgNode<ngraph::op::GetOutputElement>(op->name(), ng_cnms, 3);

  SaveNgOp(ng_op_map, op->name(), ng_nmsed_boxes);
  SaveNgOp(ng_op_map, op->name(), ng_nmsed_scores);
  SaveNgOp(ng_op_map, op->name(), ng_nmsed_classes);
  SaveNgOp(ng_op_map, op->name(), ng_valid_detections);
  return Status::OK();
}
static Status TranslateConcatV2Op(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  TF_RETURN_IF_ERROR(ValidateInputCountMin(op, 2));

  std::vector<int64> tf_concat_axis_vec;
  TF_RETURN_IF_ERROR(GetStaticInputVector(
      op, op->num_inputs() - 1, static_input_map, &tf_concat_axis_vec));

  int64 concat_axis = tf_concat_axis_vec[0];

  if (concat_axis < 0) {
    shared_ptr<ng::Node> ng_first_arg;
    TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, &ng_first_arg));

    concat_axis += int64(ng_first_arg->get_shape().size());
  }

  ng::NodeVector ng_args;

  for (int i = 0; i < op->num_inputs() - 1; i++) {
    shared_ptr<ng::Node> ng_arg;
    TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, i, &ng_arg));
    ng_args.push_back(ng_arg);
  }

  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<ng::op::Concat>(op->name(), ng_args,
                                           size_t(concat_axis)));
  return Status::OK();
}

static Status TranslateConstOp(const Node* op,
                               const std::vector<const Tensor*>&,
                               Builder::OpMap& ng_op_map) {
  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "dtype", &dtype));

  std::shared_ptr<ng::Node> ng_node;

  // For some reason the following do not work (no specialization of
  // tensorflow::checkpoint::SavedTypeTraits...)
  // case DataType::DT_UINT32:
  //   TF_RETURN_IF_ERROR(MakeConstOp<uint32>(op, ng::element::u32,
  //   &ng_node));
  //   break;
  // case DataType::DT_UINT64:
  //   TF_RETURN_IF_ERROR(MakeConstOp<uint64>(op, ng::element::u64,
  //   &ng_node));
  //   break;
  try {
    const auto& func_param = Builder::TF_NGRAPH_CONST_MAP().at(dtype);
    TF_RETURN_IF_ERROR(func_param.first(op, func_param.second, &ng_node));
  } catch (const std::out_of_range&) {
    return errors::Unimplemented("Unsupported TensorFlow data type: ",
                                 DataType_Name(dtype));
  }

  SaveNgOp(ng_op_map, op->name(), ng_node);
  return Status::OK();
}

static Status TranslateConv2DOp(const Node* op,
                                const std::vector<const Tensor*>&,
                                Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_filter;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_filter));

  std::vector<int32> tf_strides;
  std::vector<int32> tf_dilations;
  std::string tf_padding_type;
  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "dilations", &tf_dilations));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument(
        "Conv2D data format is neither NHWC nor NCHW");
  }

  bool is_nhwc = (tf_data_format == "NHWC");

  // TF Kernel Test Checks
  // Strides in the batch and depth dimension is not supported
  if (tf_strides[0] != 1 || tf_strides[is_nhwc ? 3 : 1] != 1) {
    return errors::InvalidArgument(
        "Strides in batch and depth dimensions is not supported: ",
        op->type_string());
  }

  NGRAPH_VLOG(3) << ng::join(tf_strides);
  NGRAPH_VLOG(3) << ng::join(tf_dilations);
  NGRAPH_VLOG(3) << tf_padding_type;
  NGRAPH_VLOG(3) << tf_data_format;

  ng::Strides ng_strides(2);
  ng::Strides ng_dilations(2);
  ng::Shape ng_image_shape(2);
  ng::Shape ng_kernel_shape(2);

  BatchedOpParamToNGraph(is_nhwc, tf_strides, ng_strides);
  BatchedOpParamToNGraph(is_nhwc, ng_input->get_shape(), ng_image_shape);
  BatchedOpParamToNGraph(is_nhwc, tf_dilations, ng_dilations);
  BatchToNGraph(op->name(), is_nhwc, ng_input);

  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_dilations: " << ng::join(ng_dilations);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);

  auto& ng_filter_shape = ng_filter->get_shape();
  ng_kernel_shape[0] = ng_filter_shape[0];
  ng_kernel_shape[1] = ng_filter_shape[1];
  Reshape<3, 2, 0, 1>(ng_filter);
  Builder::SetTracingInfo(op->name(), ng_filter);

  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  ng::CoordinateDiff ng_padding_below{0, 0};
  ng::CoordinateDiff ng_padding_above{0, 0};

  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_dilations, ng_padding_below,
                       ng_padding_above);

  std::shared_ptr<ng::Node> ng_conv = ConstructNgNode<ng::op::Convolution>(
      op->name(), ng_input, ng_filter, ng_strides, ng_dilations,
      ng_padding_below, ng_padding_above);

  BatchToTensorflow(op->name(), is_nhwc, ng_conv);
  SaveNgOp(ng_op_map, op->name(), ng_conv);
  return Status::OK();
}

static Status TranslateConv2DBackpropFilterOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_data_batch, ng_output_delta;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, &ng_data_batch, nullptr, &ng_output_delta));

  std::vector<int32> tf_strides;
  std::string tf_padding_type;
  std::vector<int32> tf_dilations;
  std::string tf_data_format;

  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "dilations", &tf_dilations));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument("Data format is neither NHWC nor NCHW: ",
                                   op->type_string());
  }

  NGRAPH_VLOG(3) << "tf data format" << tf_data_format;
  bool is_nhwc = (tf_data_format == "NHWC");

  // Dilations in batch and depth dimensions must be 1
  if (tf_dilations[0] != 1 || tf_dilations[is_nhwc ? 3 : 1] != 1) {
    return errors::InvalidArgument(
        "Dilations in batch and depth dimensions must be 1: ",
        op->type_string());
  }

  std::vector<int64> tf_filter_sizes;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(op, 1, static_input_map, &tf_filter_sizes));

  if (std::any_of(tf_filter_sizes.begin(), tf_filter_sizes.end(),
                  [](int32 size) { return size <= 0; })) {
    return errors::InvalidArgument("Filter sizes must be positive integers :",
                                   op->type_string());
  }

  NGRAPH_VLOG(3) << "tf filter size" << ng::join(tf_filter_sizes);
  NGRAPH_VLOG(3) << "tf filter size" << ng::join(tf_filter_sizes);
  NGRAPH_VLOG(3) << "tf strides" << ng::join(tf_strides);
  NGRAPH_VLOG(3) << "tf dilations" << ng::join(tf_dilations);
  NGRAPH_VLOG(3) << "tf padding type" << tf_padding_type;

  ng::Shape ng_filters_shape(4);
  ng::Strides ng_window_movement_strides_forward(2);
  ng::Strides ng_window_dilation_strides_forward(2);
  ng::CoordinateDiff ng_padding_below_forward{0, 0};
  ng::CoordinateDiff ng_padding_above_forward{0, 0};
  // H,W data_dilation is set to 1 , TF does not have this attribute
  ng::Strides ng_data_dilation_strides_forward(2, 1);

  // Convert inputs, args to nGraph Format
  // nGraph Data Format:
  //    nGraph Tensor           [N, C_IN, D1, ... Df]
  //    nGraph Filter           [C_OUT, C_IN, F1, ... Ff]
  //    nGraph Output Delta     [N, C_OUT, F1, ... Ff]
  //    nGraph Window Strides   [f]
  //    nGraph Window Dilations [f]
  //    nGraph Padding Below    [f]
  //    nGraph Padding Above    [f]
  //    nGraph Dilation Stride  [f]
  BatchToNGraph(op->name(), is_nhwc, ng_data_batch);
  // tf_filter shape :
  // [filter_height, filter_width, in_channels, out_channels]
  // reshape for nGraph
  ng_filters_shape = {static_cast<unsigned int>(tf_filter_sizes[3]),
                      static_cast<unsigned int>(tf_filter_sizes[2]),
                      static_cast<unsigned int>(tf_filter_sizes[0]),
                      static_cast<unsigned int>(tf_filter_sizes[1])};
  BatchToNGraph(op->name(), is_nhwc, ng_output_delta);
  BatchedOpParamToNGraph(is_nhwc, tf_strides,
                         ng_window_movement_strides_forward);
  BatchedOpParamToNGraph(is_nhwc, tf_dilations,
                         ng_window_dilation_strides_forward);
  // H, W of image/input and filter are required to figure out padding
  // arguments
  ng::Shape ng_filter_HW(2);
  ng::Shape ng_input_data_HW(2);

  auto& ng_data_batch_shape = ng_data_batch->get_shape();
  ng_input_data_HW[0] = ng_data_batch_shape[2];
  ng_input_data_HW[1] = ng_data_batch_shape[3];

  ng_filter_HW[0] = ng_filters_shape[2];
  ng_filter_HW[1] = ng_filters_shape[3];

  Builder::MakePadding(tf_padding_type, ng_input_data_HW, ng_filter_HW,
                       ng_window_movement_strides_forward,
                       ng_window_dilation_strides_forward,
                       ng_padding_below_forward, ng_padding_above_forward);

  NGRAPH_VLOG(3) << "ng input data shape" << ng::join(ng_data_batch_shape);
  NGRAPH_VLOG(3) << "ng filter shape" << ng::join(ng_filters_shape);
  NGRAPH_VLOG(3) << "ng output delta shape"
                 << ng::join(ng_output_delta->get_shape());
  NGRAPH_VLOG(3) << "ng strides"
                 << ng::join(ng_window_movement_strides_forward);
  NGRAPH_VLOG(3) << "ng dilations"
                 << ng::join(ng_window_dilation_strides_forward);
  NGRAPH_VLOG(3) << "ng padding type" << tf_padding_type;

  std::shared_ptr<ng::Node> ng_back_prop_filter =
      ConstructNgNode<ng::op::ConvolutionBackpropFilters>(
          op->name(), ng_data_batch, ng_filters_shape, ng_output_delta,
          ng_window_movement_strides_forward,
          ng_window_dilation_strides_forward, ng_padding_below_forward,
          ng_padding_above_forward, ng_data_dilation_strides_forward);

  // Reshape the output to tf format : [filter_height, filter_width,
  // in_channels, out_channels]
  Reshape<2, 3, 1, 0>(ng_back_prop_filter);
  Builder::SetTracingInfo(op->name(), ng_back_prop_filter);

  SaveNgOp(ng_op_map, op->name(), ng_back_prop_filter);
  return Status::OK();
}

static Status TranslateConv2DBackpropInputOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_filter, ng_out_backprop;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, nullptr, &ng_filter, &ng_out_backprop));

  // TODO: refactor me to be less redundant with other convolution ops
  std::vector<int32> tf_strides;
  std::vector<int32> tf_dilations;
  std::string tf_padding_type;
  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "dilations", &tf_dilations));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument(
        "Conv2DBackpropInput data format is neither NHWC nor NCHW: %s",
        tf_data_format);
  }

  std::vector<int64> tf_input_sizes;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(op, 0, static_input_map, &tf_input_sizes));

  if (std::any_of(tf_input_sizes.begin(), tf_input_sizes.end(),
                  [](int32 size) { return size <= 0; })) {
    return errors::InvalidArgument(
        "Conv2DBackpropInput input sizes must be positive integers");
  }

  bool is_nhwc = (tf_data_format == "NHWC");

  NGRAPH_VLOG(3) << ng::join(tf_strides);
  NGRAPH_VLOG(3) << ng::join(tf_dilations);
  NGRAPH_VLOG(3) << tf_padding_type;
  NGRAPH_VLOG(3) << tf_data_format;

  ng::Strides ng_strides(2);
  ng::Strides ng_dilations(2);
  ng::Shape ng_image_shape(2);
  ng::Shape ng_kernel_shape(2);
  ng::Shape ng_batch_shape(4);

  BatchedOpParamToNGraph(is_nhwc, tf_strides, ng_strides);
  BatchedOpParamToNGraph(is_nhwc, tf_input_sizes, ng_image_shape);
  BatchedOpParamToNGraph(is_nhwc, tf_dilations, ng_dilations);
  BatchToNGraph(op->name(), is_nhwc, ng_out_backprop);
  if (is_nhwc) {
    ng_batch_shape = {static_cast<unsigned long>(tf_input_sizes[0]),
                      static_cast<unsigned long>(tf_input_sizes[3]),
                      static_cast<unsigned long>(tf_input_sizes[1]),
                      static_cast<unsigned long>(tf_input_sizes[2])};
  } else {
    ng_batch_shape = {static_cast<unsigned long>(tf_input_sizes[0]),
                      static_cast<unsigned long>(tf_input_sizes[1]),
                      static_cast<unsigned long>(tf_input_sizes[2]),
                      static_cast<unsigned long>(tf_input_sizes[3])};
  }

  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_dilations: " << ng::join(ng_dilations);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);

  auto& ng_filter_shape = ng_filter->get_shape();
  ng_kernel_shape[0] = ng_filter_shape[0];
  ng_kernel_shape[1] = ng_filter_shape[1];
  Reshape<3, 2, 0, 1>(ng_filter);
  Builder::SetTracingInfo(op->name(), ng_filter);

  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  ng::CoordinateDiff ng_padding_below{0, 0};
  ng::CoordinateDiff ng_padding_above{0, 0};

  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_dilations, ng_padding_below,
                       ng_padding_above);

  std::shared_ptr<ng::Node> ng_data =
      ConstructNgNode<ng::op::ConvolutionBackpropData>(
          op->name(), ng_batch_shape, ng_filter, ng_out_backprop, ng_strides,
          ng_dilations, ng_padding_below, ng_padding_above,
          ng::Strides(ng_batch_shape.size() - 2, 1));

  BatchToTensorflow(op->name(), is_nhwc, ng_data);

  SaveNgOp(ng_op_map, op->name(), ng_data);
  return Status::OK();
}

// Translate Conv3D Op
static Status TranslateConv3DOp(const Node* op,
                                const std::vector<const Tensor*>&,
                                Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_filter;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_filter));

  std::vector<int32> tf_strides;
  std::vector<int32> tf_dilations;
  std::string tf_padding_type;
  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "dilations", &tf_dilations));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if (tf_data_format != "NDHWC" && tf_data_format != "NCDHW") {
    return errors::InvalidArgument(
        "Conv3D data format is neither NDHWC nor NCDHW");
  }

  bool is_ndhwc = (tf_data_format == "NDHWC");

  // TODO: in 3D
  // TF Kernel Test Checks
  // // Strides in the batch and depth dimension is not supported
  // if (tf_strides[0] != 1 || tf_strides[is_nhwc ? 3 : 1] != 1) {
  //   return errors::InvalidArgument(
  //       "Strides in batch and depth dimensions is not supported: ",
  //       op->type_string());
  // }

  NGRAPH_VLOG(3) << ng::join(tf_strides);
  NGRAPH_VLOG(3) << ng::join(tf_dilations);
  NGRAPH_VLOG(3) << tf_padding_type;
  NGRAPH_VLOG(3) << tf_data_format;

  ng::Strides ng_strides(3);
  ng::Strides ng_dilations(3);
  ng::Shape ng_image_shape(3);
  ng::Shape ng_kernel_shape(3);

  BatchedOpParam3DToNGraph(is_ndhwc, tf_strides, ng_strides);
  BatchedOpParam3DToNGraph(is_ndhwc, ng_input->get_shape(), ng_image_shape);
  BatchedOpParam3DToNGraph(is_ndhwc, tf_dilations, ng_dilations);
  BatchToNGraph3D(op->name(), is_ndhwc, ng_input);

  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_dilations: " << ng::join(ng_dilations);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);

  auto& ng_filter_shape = ng_filter->get_shape();
  ng_kernel_shape[0] = ng_filter_shape[0];
  ng_kernel_shape[1] = ng_filter_shape[1];
  ng_kernel_shape[2] = ng_filter_shape[2];
  Reshape3D<4, 3, 0, 1, 2>(ng_filter);
  Builder::SetTracingInfo(op->name(), ng_filter);

  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  ng::CoordinateDiff ng_padding_below{0, 0, 0};
  ng::CoordinateDiff ng_padding_above{0, 0, 0};

  Builder::MakePadding3D(tf_padding_type, ng_image_shape, ng_kernel_shape,
                         ng_strides, ng_dilations, ng_padding_below,
                         ng_padding_above);

  std::shared_ptr<ng::Node> ng_conv = ConstructNgNode<ng::op::Convolution>(
      op->name(), ng_input, ng_filter, ng_strides, ng_dilations,
      ng_padding_below, ng_padding_above);

  BatchToTensorflow3D(op->name(), is_ndhwc, ng_conv);
  SaveNgOp(ng_op_map, op->name(), ng_conv);
  return Status::OK();
}

// Translate DepthToSpace op
static Status TranslateDepthToSpaceOp(const Node* op,
                                      const std::vector<const Tensor*>&,
                                      Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  // Get the attributes
  int64 block_size;
  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "block_size", &block_size));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  ng::Shape input_shape = ng_input->get_shape();
  std::map<std::string, int> format_to_int_map = {
      {"NHWC", 0}, {"NCHW", 1}, {"NCHW_VECT_C", 1}};

  int channel_dimension;
  int num_spatial_dimensions = 2;  // H, W are spatial dimensions

  switch (format_to_int_map[tf_data_format]) {
    // NHWC
    case 0:
      channel_dimension = 3;
      break;
    // NCHW
    case 1:
      channel_dimension = 1;
      break;
    // NCHW_VEC_C
    case 2:
      return errors::InvalidArgument(
          "NCHW_VECT_C is not supported in DepthToSpace for now");
    default:
      return errors::InvalidArgument(
          "DepthToSpace supported data format is NCHW, NHWC, or NCHW_VECT_C");
  }

  // Error checking : depth must be divisible by square of the block_size
  if (input_shape[channel_dimension] % (block_size * block_size) != 0) {
    return errors::InvalidArgument(
        "Input tensor's channel dimension ,", input_shape[channel_dimension],
        " is not divisible by square of the block_size ", block_size);
  }

  ng::AxisVector ng_reshape_shape;
  ng::AxisVector ng_transpose_permutation;
  ng::AxisVector ng_output_shape;

  switch (format_to_int_map[tf_data_format]) {
    // NHWC
    case 0: {
      // ng_reshape_shape = [batch_size,
      //                     height,
      //                     width,
      //                     block_size,
      //                     block_size,
      //                     channel / (block_size * block_size)]
      ng_reshape_shape.push_back(input_shape[0]);
      for (int i = 0; i < num_spatial_dimensions; i++) {
        ng_reshape_shape.push_back(input_shape[i + 1]);
      }

      int64 num_blocks = 1;
      for (int i = 0; i < num_spatial_dimensions; i++) {
        ng_reshape_shape.push_back(block_size);
        num_blocks *= block_size;
      }
      ng_reshape_shape.push_back(input_shape[channel_dimension] / num_blocks);

      // ng_transpose_shape = [batch_size,
      //                       height,
      //                       block_size,
      //                       width,
      //                       block_size,
      //                       channel / (block_size * block_size)]
      ng_transpose_permutation.push_back(0);
      for (int i = 0; i < num_spatial_dimensions; i++) {
        ng_transpose_permutation.push_back(i + 1);
        ng_transpose_permutation.push_back(i + 1 + num_spatial_dimensions);
      }
      ng_transpose_permutation.push_back(channel_dimension +
                                         num_spatial_dimensions);

      // ng_output_shape = [batch_size,
      //                    height * block_size,
      //                    width * block_size,
      //                    channel / (block_size * block_size)]
      ng_output_shape.push_back(input_shape[0]);
      for (int i = 0; i < num_spatial_dimensions; i++) {
        ng_output_shape.push_back(input_shape[i + 1] * block_size);
      }
      ng_output_shape.push_back(input_shape[channel_dimension] / num_blocks);
      break;
    }  // end of case NHWC

    // NCHW
    case 1: {
      // ng_reshape_shape = [batch_size,
      //                     block_size,
      //                     block_size,
      //                     channel / (block_size * block_size),
      //                     height,
      //                     width]
      int64 num_blocks = 1;
      ng_reshape_shape.push_back(input_shape[0]);  // N dimension
      for (int i = 0; i < num_spatial_dimensions; i++) {
        ng_reshape_shape.push_back(block_size);
        num_blocks *= block_size;
      }
      ng_reshape_shape.push_back(input_shape[channel_dimension] / num_blocks);

      for (int i = 0; i < num_spatial_dimensions; i++) {
        ng_reshape_shape.push_back(input_shape[i + 2]);
      }

      // ng_transpose_shape = [batch_size,
      //                       channel / (block_size * block_size)
      //                       height,
      //                       block_size,
      //                       width,
      //                       block_size]
      ng_transpose_permutation.push_back(0);
      ng_transpose_permutation.push_back(1 + num_spatial_dimensions);
      for (int i = 0; i < num_spatial_dimensions; i++) {
        ng_transpose_permutation.push_back(i + 2 + num_spatial_dimensions);
        ng_transpose_permutation.push_back(i + 1);
      }

      // ng_output_shape = [batch_size,
      //                    channel / (block_size * block_size)
      //                    height * block_size,
      //                    width * block_size]
      ng_output_shape.push_back(input_shape[0]);
      ng_output_shape.push_back(input_shape[channel_dimension] / num_blocks);
      for (int i = 0; i < num_spatial_dimensions; i++) {
        ng_output_shape.push_back(input_shape[i + 2] * block_size);
      }
      break;
    }  // end of case NCHW
  }

  ng::AxisVector ng_axis_order(input_shape.size());
  std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);
  auto reshaped = ConstructNgNode<ng::op::Reshape>(
      op->name(), ng_input, ng_axis_order, ng_reshape_shape);

  auto transposed =
      ng::builder::numpy_transpose(reshaped, ng_transpose_permutation);
  Builder::SetTracingInfo(op->name(), transposed);

  ng::AxisVector ng_axis_order_second_reshape(transposed->get_shape().size());
  std::iota(ng_axis_order_second_reshape.begin(),
            ng_axis_order_second_reshape.end(), 0);
  auto final_reshape = ConstructNgNode<ng::op::Reshape>(
      op->name(), transposed, ng_axis_order_second_reshape, ng_output_shape);
  SaveNgOp(ng_op_map, op->name(), final_reshape);

  return Status::OK();
}

static Status TranslateDepthwiseConv2dNativeOp(
    const Node* op, const std::vector<const Tensor*>&,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_filter;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_filter));

  std::vector<int32> tf_strides;
  std::vector<int32> tf_dilations;
  std::string tf_padding_type;
  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "dilations", &tf_dilations));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument(
        "DepthwiseConv2D data format is neither NHWC nor NCHW");
  }

  bool is_nhwc = (tf_data_format == "NHWC");

  NGRAPH_VLOG(3) << ng::join(tf_strides);
  NGRAPH_VLOG(3) << ng::join(tf_dilations);
  NGRAPH_VLOG(3) << tf_padding_type;
  NGRAPH_VLOG(3) << tf_data_format;

  ng::Strides ng_strides(2);
  ng::Strides ng_dilations(2);
  ng::Shape ng_image_shape(2);
  ng::Shape ng_kernel_shape(2);

  BatchedOpParamToNGraph(is_nhwc, ng_input->get_shape(), ng_image_shape);
  BatchedOpParamToNGraph(is_nhwc, tf_strides, ng_strides);
  BatchedOpParamToNGraph(is_nhwc, tf_dilations, ng_dilations);
  BatchToNGraph(op->name(), is_nhwc, ng_input);

  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_dilations: " << ng::join(ng_dilations);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);

  auto& ng_filter_shape = ng_filter->get_shape();
  ng_kernel_shape[0] = ng_filter_shape[0];
  ng_kernel_shape[1] = ng_filter_shape[1];
  Reshape<3, 2, 0, 1>(ng_filter);
  Builder::SetTracingInfo(op->name(), ng_filter);

  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  ng::CoordinateDiff ng_padding_below{0, 0};
  ng::CoordinateDiff ng_padding_above{0, 0};

  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_dilations, ng_padding_below,
                       ng_padding_above);

  // ng input shape is NCHW
  auto& input_shape = ng_input->get_shape();
  // ng filter shape is OIHW
  auto& filter_shape = ng_filter->get_shape();
  ng::NodeVector ng_args;

  for (size_t i = 0; i < input_shape[1]; i++) {
    const std::vector<size_t> lower_bound{0, i, 0, 0};
    const std::vector<size_t> upper_bound{input_shape[0], i + 1, input_shape[2],
                                          input_shape[3]};
    auto ng_sliced_input = ConstructNgNode<ng::op::Slice>(
        op->name(), ng_input, lower_bound, upper_bound);

    const std::vector<size_t> f_lower_bound{0, i, 0, 0};
    const std::vector<size_t> f_upper_bound{filter_shape[0], i + 1,
                                            filter_shape[2], filter_shape[3]};
    auto ng_sliced_filter = ConstructNgNode<ng::op::Slice>(
        op->name(), ng_filter, f_lower_bound, f_upper_bound);

    NGRAPH_VLOG(3) << "depthwise conv 2d.";
    NGRAPH_VLOG(3) << "sliced shape " << ng::join(ng_sliced_input->get_shape());
    NGRAPH_VLOG(3) << "filter shape "
                   << ng::join(ng_sliced_filter->get_shape());

    auto ng_conv = ConstructNgNode<ng::op::Convolution>(
        op->name(), ng_sliced_input, ng_sliced_filter, ng_strides, ng_dilations,
        ng_padding_below, ng_padding_above);
    ng_args.push_back(ng_conv);
  }

  size_t ng_concatenation_axis = 1;  // channel axis
  std::shared_ptr<ng::Node> ng_concat = ConstructNgNode<ng::op::Concat>(
      op->name(), ng_args, ng_concatenation_axis);

  BatchToTensorflow(op->name(), is_nhwc, ng_concat);
  SaveNgOp(ng_op_map, op->name(), ng_concat);
  return Status::OK();
}

static Status TranslateExpandDimsOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_dim;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_dim));

  std::vector<int64> dim_vec;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &dim_vec));

  if (dim_vec.size() != 1) {
    return errors::InvalidArgument(
        "The size of argument dim is not 1 for ExpandDims");
  }

  auto& shape = ng_input->get_shape();
  auto shape_size = shape.size();
  if (dim_vec[0] < 0) {
    // allow range [-rank(input) - 1, rank(input)]
    // where -1 append new axis at the end
    dim_vec[0] = shape_size + dim_vec[0] + 1;
  }
  auto out_shape = shape;
  out_shape.insert(out_shape.begin() + size_t(dim_vec[0]), 1);
  std::vector<size_t> shape_dimensions(shape.size());
  std::iota(shape_dimensions.begin(), shape_dimensions.end(), 0);
  std::shared_ptr<ng::Node> ng_expand_dim = ConstructNgNode<ng::op::Reshape>(
      op->name(), ng_input, shape_dimensions, out_shape);

  SaveNgOp(ng_op_map, op->name(), ng_expand_dim);
  return Status::OK();
}

static Status TranslateFillOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_value;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, nullptr, &ng_value));

  std::vector<int64> dims_vec;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 0, static_input_map, &dims_vec));

  ng::Shape ng_output_shape(dims_vec.size());
  ng::AxisSet ng_axis_set;
  for (size_t i = 0; i < dims_vec.size(); ++i) {
    ng_output_shape[i] = dims_vec[i];
    ng_axis_set.insert(i);
  }
  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<ng::op::Broadcast>(op->name(), ng_value,
                                              ng_output_shape, ng_axis_set));
  return Status::OK();
}

static Status TranslateFloorDivOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  auto ng_floordiv = [&op](std::shared_ptr<ng::Node> ng_input1,
                           std::shared_ptr<ng::Node> ng_input2) {
    return ConstructNgNode<ng::op::Floor>(
        op->name(),
        ConstructNgNode<ng::op::Divide>(op->name(), ng_input1, ng_input2));
  };
  return TranslateBinaryOp(op, static_input_map, ng_op_map, ng_floordiv);
}

static Status TranslateFloorModOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  auto ng_floormod = [&op](std::shared_ptr<ng::Node> ng_input1,
                           std::shared_ptr<ng::Node> ng_input2) {
    auto floordiv = ConstructNgNode<ng::op::Floor>(
        op->name(),
        ConstructNgNode<ng::op::Divide>(op->name(), ng_input1, ng_input2));
    return ConstructNgNode<ng::op::Subtract>(
        op->name(), ng_input1,
        ConstructNgNode<ng::op::Multiply>(op->name(), floordiv, ng_input2));
  };
  return TranslateBinaryOp(op, static_input_map, ng_op_map, ng_floormod);
}

static Status TranslateFusedBatchNormOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  bool tf_is_training;
  if (GetNodeAttr(op->attrs(), "is_training", &tf_is_training) !=
      Status::OK()) {
    NGRAPH_VLOG(3) << "is_training attribute not present, setting to true";
    tf_is_training = true;
  }

  NGRAPH_VLOG(3) << "is_training: " << tf_is_training;

  shared_ptr<ng::Node> ng_input, ng_scale, ng_offset, ng_mean, ng_variance;
  bool is_v3 = op->type_string() == "FusedBatchNormV3";
  if (tf_is_training) {
    TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, &ng_input));
    TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 1, &ng_scale));
    TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 2, &ng_offset));
  } else {
    TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_scale,
                                     &ng_offset, &ng_mean, &ng_variance));
  }

  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument(
        "Conv2D data format is neither NHWC nor NCHW");
  }

  bool is_nhwc = (tf_data_format == "NHWC");

  NGRAPH_VLOG(3) << "data_format: " << tf_data_format;

  float tf_epsilon;
  if (GetNodeAttr(op->attrs(), "epsilon", &tf_epsilon) != Status::OK()) {
    NGRAPH_VLOG(3) << "epsilon attribute not present, setting to 0.0001";
    // TensorFlow default
    tf_epsilon = 0.0001;
  }

  NGRAPH_VLOG(3) << "epsilon: " << tf_epsilon;

  BatchToNGraph(op->name(), is_nhwc, ng_input);

  std::shared_ptr<ng::Node> ng_batch_norm;

  if (tf_is_training) {
    ng_batch_norm = ConstructNgNode<ng::op::BatchNormTraining>(
        op->name(), tf_epsilon, ng_scale, ng_offset, ng_input);

    shared_ptr<ngraph::Node> ng_y, ng_mean, ng_variance;
    ng_y =
        ConstructNgNode<ng::op::GetOutputElement>(op->name(), ng_batch_norm, 0);
    ng_mean =
        ConstructNgNode<ng::op::GetOutputElement>(op->name(), ng_batch_norm, 1);
    ng_variance =
        ConstructNgNode<ng::op::GetOutputElement>(op->name(), ng_batch_norm, 2);
    // This is for Bessel's correction in ng_variance:
    int ng_input_size = ng::shape_size(ng_input->get_shape());
    int num_channels = ng::shape_size(ng_variance->get_shape());
    int sample_size = ng_input_size / num_channels;
    int sample_size_minus_one = sample_size > 1 ? (sample_size - 1) : 1;
    float factor = float(sample_size) / float(sample_size_minus_one);
    std::vector<float> Bessel_factor(num_channels, factor);
    auto Bessel_scale = ConstructNgNode<ng::op::Constant>(
        op->name(), ng_variance->get_element_type(), ng_variance->get_shape(),
        Bessel_factor);
    auto variance = ConstructNgNode<ng::op::Multiply>(op->name(), ng_variance,
                                                      Bessel_scale);

    BatchToTensorflow(op->name(), is_nhwc, ng_y);

    SaveNgOp(ng_op_map, op->name(), ng_y);
    SaveNgOp(ng_op_map, op->name(), ng_mean);
    SaveNgOp(ng_op_map, op->name(), variance);
    // Output reserve_space_1: A 1D Tensor for the computed batch mean, to be
    // reused in the gradient computation.
    SaveNgOp(ng_op_map, op->name(), ng_mean);
    // Output reserve_space_2: A 1D Tensor for the computed batch variance
    //(inverted variance in the cuDNN case), to be reused in the gradient
    // computation.
    SaveNgOp(ng_op_map, op->name(), ng_variance);
    if (is_v3) {
      // FusedBatchNormV3 has 6 outputs (reserve_space_3)
      shared_ptr<ng::Node> ng_reserved_3 =
          ConstructNgNode<ngraph::op::Constant>(
              op->name(), ng_mean->get_element_type(), ng::Shape{},
              std::vector<std::string>{""});
      SaveNgOp(ng_op_map, op->name(), ng_reserved_3);
    }
  } else {
    ng_batch_norm = ConstructNgNode<ng::op::BatchNormInference>(
        op->name(), tf_epsilon, ng_scale, ng_offset, ng_input, ng_mean,
        ng_variance);
    BatchToTensorflow(op->name(), is_nhwc, ng_batch_norm);
    SaveNgOp(ng_op_map, op->name(), ng_batch_norm);
    if (is_v3) {
      SaveNgOp(ng_op_map, op->name(), ng_mean);
      SaveNgOp(ng_op_map, op->name(), ng_variance);
      SaveNgOp(ng_op_map, op->name(), ng_mean);
      SaveNgOp(ng_op_map, op->name(), ng_variance);
      // FusedBatchNormV3 has 6 outputs (reserve_space_3)
      shared_ptr<ng::Node> ng_reserved_3 =
          ConstructNgNode<ngraph::op::Constant>(
              op->name(), ng_mean->get_element_type(), ng::Shape{},
              std::vector<std::string>{""});
      SaveNgOp(ng_op_map, op->name(), ng_reserved_3);
    }
  }

  return Status::OK();
}

static Status TranslateFusedBatchNormGradOp(const Node* op,
                                            const std::vector<const Tensor*>&,
                                            Builder::OpMap& ng_op_map) {
  bool is_v3 = op->type_string() == "FusedBatchNormGradV3";
  TF_RETURN_IF_ERROR(ValidateInputCount(op, is_v3 ? 6 : 5));

  bool tf_is_training;
  // We only support is_training=true case. We marked rejection for the case
  // is_training=false.
  if (GetNodeAttr(op->attrs(), "is_training", &tf_is_training) !=
      Status::OK()) {
    NGRAPH_VLOG(3) << "is_training attribute not present, setting to true";
    tf_is_training = true;
  }

  NGRAPH_VLOG(3) << "is_training: " << tf_is_training;

  shared_ptr<ng::Node> ng_delta;
  shared_ptr<ng::Node> ng_input;
  shared_ptr<ng::Node> ng_scale;
  shared_ptr<ng::Node> ng_mean;
  shared_ptr<ng::Node> ng_variance;
  if (is_v3) {
    TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_delta, &ng_input,
                                     &ng_scale, &ng_mean, &ng_variance,
                                     nullptr));
  } else {
    TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_delta, &ng_input,
                                     &ng_scale, &ng_mean, &ng_variance));
  }

  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument(
        "FusedBatchnormGrad data format is neither NHWC nor NCHW");
  }

  bool is_nhwc = (tf_data_format == "NHWC");

  NGRAPH_VLOG(3) << "data_format: " << tf_data_format;

  float tf_epsilon;
  if (GetNodeAttr(op->attrs(), "epsilon", &tf_epsilon) != Status::OK()) {
    NGRAPH_VLOG(3) << "epsilon attribute not present, setting to 0.0001";
    tf_epsilon = 0.0001;
  }

  NGRAPH_VLOG(3) << "epsilon: " << tf_epsilon;

  // TODO: We are temporarily supplying a fake value for beta here
  // (all zero, same shape/et as scale/gamma), because Tensorflow does not give
  // beta to us. This should work because nGraph should not actually use beta.
  // The nGraph op may change to discard this parameter. Update this when nGraph
  // does.
  shared_ptr<ng::Node> ng_beta = ConstructNgNode<ngraph::op::Constant>(
      op->name(), ng_scale->get_element_type(), ng_scale->get_shape(),
      std::vector<std::string>{ng::shape_size(ng_scale->get_shape()), "0"});

  BatchToNGraph(op->name(), is_nhwc, ng_input);
  BatchToNGraph(op->name(), is_nhwc, ng_delta);

  std::shared_ptr<ng::Node> ng_batch_norm_backprop;

  ng_batch_norm_backprop = ConstructNgNode<ng::op::BatchNormTrainingBackprop>(
      op->name(), tf_epsilon, ng_scale, ng_beta, ng_input, ng_mean, ng_variance,
      ng_delta);

  shared_ptr<ngraph::Node> ng_input_delta_op =
      ConstructNgNode<ng::op::GetOutputElement>(op->name(),
                                                ng_batch_norm_backprop, 0);
  shared_ptr<ngraph::Node> ng_scale_delta_op =
      ConstructNgNode<ng::op::GetOutputElement>(op->name(),
                                                ng_batch_norm_backprop, 1);
  shared_ptr<ngraph::Node> ng_beta_delta_op =
      ConstructNgNode<ng::op::GetOutputElement>(op->name(),
                                                ng_batch_norm_backprop, 2);

  BatchToTensorflow(op->name(), is_nhwc, ng_input_delta_op);

  SaveNgOp(ng_op_map, op->name(), ng_input_delta_op);
  SaveNgOp(ng_op_map, op->name(), ng_scale_delta_op);
  SaveNgOp(ng_op_map, op->name(), ng_beta_delta_op);
  // Output reserve_space_3: Unused placeholder to match the mean input
  // in FusedBatchNorm.
  std::shared_ptr<ng::Node> output_mean = ConstructNgNode<ngraph::op::Constant>(
      op->name(), ng_mean->get_element_type(), ng::Shape{},
      std::vector<std::string>{""});
  SaveNgOp(ng_op_map, op->name(), output_mean);
  // Output reserve_space_4: Unused placeholder to match the variance input
  // in FusedBatchNorm.
  std::shared_ptr<ng::Node> output_variance =
      ConstructNgNode<ngraph::op::Constant>(
          op->name(), ng_variance->get_element_type(), ng::Shape{},
          std::vector<std::string>{""});
  SaveNgOp(ng_op_map, op->name(), output_variance);

  return Status::OK();
}

static Status TranslateGatherNdOp(const Node* op,
                                  const std::vector<const Tensor*>&,
                                  Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_params;
  shared_ptr<ng::Node> ng_indices;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_params, &ng_indices));

  auto ng_params_shape = ng_params->get_shape();
  size_t ng_params_rank = ng_params_shape.size();
  size_t ng_indices_rank = ng_indices->get_shape().size();

  for (size_t i = 0; i < ng_params_rank; i++) {
    if (ng_params_shape[i] == 0) {
      return errors::InvalidArgument(
          "Requested more than 0 entries, but params is empty.  Params shape: "
          "[",
          ng::join(ng_params_shape, ","), "]");
    }
  }

  if ((ng_indices_rank - 1) > ng_params_rank) {
    return errors::InvalidArgument(
        "The last dimension of indices can be at most the rank of params");
  }

  SaveNgOp(ng_op_map, op->name(), ConstructNgNode<ng::op::GatherND>(
                                      op->name(), ng_params, ng_indices));

  return Status::OK();
}

static Status TranslateFusedMatMulOp(const Node* op,
                                     const std::vector<const Tensor*>&,
                                     Builder::OpMap& ng_op_map) {
  int num_args;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "num_args", &num_args));

  std::vector<string> fused_ops;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "fused_ops", &fused_ops));

  auto CreateNgDot = [&](shared_ptr<ng::Node>& ng_lhs,
                         shared_ptr<ng::Node>& ng_rhs,
                         shared_ptr<ng::Node>& ng_dot) {

    // Transpose arguments if requested.
    bool transpose_a = false;
    bool transpose_b = false;

    if (GetNodeAttr(op->attrs(), "transpose_a", &transpose_a) == Status::OK() &&
        transpose_a) {
      ng_lhs = ng::builder::numpy_transpose(ng_lhs, ng::AxisVector{1, 0});
      Builder::SetTracingInfo(op->name(), ng_lhs);
    }
    if (GetNodeAttr(op->attrs(), "transpose_b", &transpose_b) == Status::OK() &&
        transpose_b) {
      ng_rhs = ng::builder::numpy_transpose(ng_rhs, ng::AxisVector{1, 0});
      Builder::SetTracingInfo(op->name(), ng_rhs);
    }

    // The default axis count for nGraph's Dot op is 1, which is just what
    // we need here.
    ng_dot = ConstructNgNode<ngraph::op::Dot>(op->name(), ng_lhs, ng_rhs);

    return Status::OK();
  };

  shared_ptr<ng::Node> ng_lhs, ng_rhs, ng_bias, ng_dot;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_lhs, &ng_rhs, &ng_bias));
  TF_RETURN_IF_ERROR(CreateNgDot(ng_lhs, ng_rhs, ng_dot));

  auto ng_dot_shape = ng_dot->get_shape();
  auto ng_bias_shape = ng_bias->get_shape();

  if (ng_bias_shape.size() != 1) {
    return errors::InvalidArgument(
        "Bias argument to BiasAdd does not have one dimension");
  }

  ng::AxisSet ng_broadcast_axes;

  // TODO : _FusedMatMul doesn't have data_format attributes, insert broadcast
  // axes as if it's NHWC for now.
  for (size_t i = 0; i < ng_dot_shape.size() - 1; i++) {
    ng_broadcast_axes.insert(i);
  }

  auto ng_bias_broadcasted = ConstructNgNode<ng::op::Broadcast>(
      op->name(), ng_bias, ng_dot_shape, ng_broadcast_axes);

  auto ng_add =
      ConstructNgNode<ng::op::Add>(op->name(), ng_dot, ng_bias_broadcasted);
  if (fused_ops.size() == 1) {  // Only fusing BiasAdd
    SaveNgOp(ng_op_map, op->name(), ng_add);
  } else if (fused_ops.size() == 2) {  // Also has activation
    if (fused_ops[1] == "Relu") {
      SaveNgOp(ng_op_map, op->name(),
               ConstructNgNode<ng::op::Relu>(op->name(), ng_add));
    } else if (fused_ops[1] == "Relu6") {
      // TODO fill
      auto constant_6 = ConstructNgNode<ng::op::Constant>(
          op->name(), ng_add->get_element_type(), ng_add->get_shape(),
          std::vector<std::string>(ng::shape_size(ng_add->get_shape()), "6"));
      auto relu6_op = ConstructNgNode<ng::op::Minimum>(
          op->name(), ConstructNgNode<ng::op::Relu>(op->name(), ng_add),
          constant_6);
      SaveNgOp(ng_op_map, op->name(), relu6_op);
    } else {
      return errors::Internal(
          "Expected activation to be Relu or Relu6 but got ", fused_ops[1]);
    }
  } else {
    // Adding this here to catch future changes in _FusedMatMul
    return errors::Internal("Unsupported combination");
  }

  return Status::OK();
}

static Status TranslateGatherV2Op(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_input_coords;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, &ng_input, &ng_input_coords, nullptr));

  std::vector<int64> tf_axis;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 2, static_input_map, &tf_axis));

  if (tf_axis.size() > 1) {
    return errors::Internal("Found axis in GatherV2 op (", op->name(),
                            ") translation to be non scalar, of size ",
                            tf_axis.size());
  }

  std::string backend_name;
  TF_RETURN_IF_ERROR(ngraph_bridge::GetNodeBackend(op, &backend_name));

  // split and check the first part only, since the node attribute contains
  // the full backend creation string
  auto config_map = BackendManager::GetBackendAttributeValues(backend_name);
  if (config_map.at("ngraph_backend") != "NNPI") {
    return errors::Internal("In translating GatherV2 op ", op->name(),
                            " found requested backend ", backend_name,
                            " which is unsupported");
  }

  ng::runtime::Backend* backend = BackendManager::GetBackend(backend_name);

  // Negative axis is supported. Accounting for that
  auto ng_input_shape = ng_input->get_shape();
  size_t ng_input_rank = ng_input_shape.size();
  int axis;
  if (tf_axis[0] >= 0) {
    axis = tf_axis[0];
  } else {
    axis = tf_axis[0] + ng_input_rank;
  }
  if (axis < 0 || axis >= ng_input_rank) {
    return errors::InvalidArgument("Expected axis in the range [-",
                                   ng_input_rank, ", ", ng_input_rank,
                                   "), but got ", tf_axis[0]);
  }

  shared_ptr<ng::Node> ng_gather =
      backend->get_backend_op("Gather", &ng_input, &ng_input_coords, &axis);
  if (ng_gather == nullptr) {
    return errors::Internal("In translating GatherV2 op ", op->name(),
                            " backend could not return valid ngraph node");
  }
  Builder::SetTracingInfo(op->name(), ng_gather);
  SaveNgOp(ng_op_map, op->name(), ng_gather);

  return Status::OK();
}

static Status TranslateFusedConv2DOp(const Node* op,
                                     const std::vector<const Tensor*>&,
                                     Builder::OpMap& ng_op_map) {
  int num_args;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "num_args", &num_args));

  std::vector<string> fused_ops;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "fused_ops", &fused_ops));

  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));
  bool is_nhwc = (tf_data_format == "NHWC");

  auto CreateNgConv = [&](shared_ptr<ng::Node>& ng_input,
                          shared_ptr<ng::Node>& ng_filter,
                          shared_ptr<ng::Node>& ng_conv) {
    std::vector<int32> tf_strides;
    std::vector<int32> tf_dilations;
    std::string tf_padding_type;
    TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
    TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "dilations", &tf_dilations));
    TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));

    if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
      return errors::InvalidArgument(
          "Conv2D data format is neither NHWC nor NCHW");
    }

    // TF Kernel Test Checks
    // Strides in the batch and depth dimension is not supported
    if (tf_strides[0] != 1 || tf_strides[is_nhwc ? 3 : 1] != 1) {
      return errors::InvalidArgument(
          "Strides in batch and depth dimensions is not supported: ",
          op->type_string());
    }

    NGRAPH_VLOG(3) << ng::join(tf_strides);
    NGRAPH_VLOG(3) << ng::join(tf_dilations);
    NGRAPH_VLOG(3) << tf_padding_type;
    NGRAPH_VLOG(3) << tf_data_format;

    ng::Strides ng_strides(2);
    ng::Strides ng_dilations(2);
    ng::Shape ng_image_shape(2);
    ng::Shape ng_kernel_shape(2);

    BatchedOpParamToNGraph(is_nhwc, tf_strides, ng_strides);
    BatchedOpParamToNGraph(is_nhwc, ng_input->get_shape(), ng_image_shape);
    BatchedOpParamToNGraph(is_nhwc, tf_dilations, ng_dilations);
    BatchToNGraph(op->name(), is_nhwc, ng_input);

    NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
    NGRAPH_VLOG(3) << "ng_dilations: " << ng::join(ng_dilations);
    NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);

    auto& ng_filter_shape = ng_filter->get_shape();
    ng_kernel_shape[0] = ng_filter_shape[0];
    ng_kernel_shape[1] = ng_filter_shape[1];
    Reshape<3, 2, 0, 1>(ng_filter);
    Builder::SetTracingInfo(op->name(), ng_filter);

    NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

    ng::CoordinateDiff ng_padding_below{0, 0};
    ng::CoordinateDiff ng_padding_above{0, 0};

    Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                         ng_strides, ng_dilations, ng_padding_below,
                         ng_padding_above);

    ng_conv = ConstructNgNode<ng::op::Convolution>(
        op->name() + "_FusedConv2D_Conv", ng_input, ng_filter, ng_strides,
        ng_dilations, ng_padding_below, ng_padding_above);

    return Status::OK();
  };

  auto create_relu6 = [](const string& op_name,
                         const shared_ptr<ng::Node>& ng_node) {
    auto constant_6 = ConstructNgNode<ng::op::Constant>(
        op_name, ng_node->get_element_type(), ng_node->get_shape(),
        std::vector<std::string>(ng::shape_size(ng_node->get_shape()), "6"));
    auto relu6_op = ConstructNgNode<ng::op::Minimum>(
        op_name,
        ConstructNgNode<ng::op::Relu>(op_name + "_FusedConv2D_Relu", ng_node),
        constant_6);
    return relu6_op;
  };

  if (VecStrCmp(fused_ops, {"BiasAdd"}) ||
      VecStrCmp(fused_ops, {"BiasAdd", "Relu"}) ||
      VecStrCmp(fused_ops, {"BiasAdd", "Relu6"})) {
    if (num_args != 1) {
      return errors::InvalidArgument(
          "FusedConv2DBiasAdd has incompatible num_args");
    }

    shared_ptr<ng::Node> ng_input, ng_filter, ng_bias, ng_conv;
    TF_RETURN_IF_ERROR(
        GetInputNodes(ng_op_map, op, &ng_input, &ng_filter, &ng_bias));

    TF_RETURN_IF_ERROR(CreateNgConv(ng_input, ng_filter, ng_conv));

    BatchToTensorflow(op->name(), is_nhwc, ng_conv);

    auto ng_conv_shape = ng_conv->get_shape();
    auto ng_bias_shape = ng_bias->get_shape();
    if (ng_bias_shape.size() != 1) {
      return errors::InvalidArgument(
          "Bias argument to BiasAdd does not have one dimension");
    }

    ng::AxisSet ng_broadcast_axes;

    if (is_nhwc) {
      for (size_t i = 0; i < ng_conv_shape.size() - 1; i++) {
        ng_broadcast_axes.insert(i);
      }
    } else {
      for (size_t i = 0; i < ng_conv_shape.size(); i++) {
        if (i != 1) {
          ng_broadcast_axes.insert(i);
        }
      }
    }

    auto ng_bias_broadcasted = ConstructNgNode<ng::op::Broadcast>(
        op->name() + "_FusedConv2D_BiasAdd", ng_bias, ng_conv_shape,
        ng_broadcast_axes);
    auto ng_add = ConstructNgNode<ng::op::Add>(
        op->name() + "_FusedConv2D_BiasAdd", ng_conv, ng_bias_broadcasted);

    if (VecStrCmp(fused_ops, {"BiasAdd", "Relu"})) {
      SaveNgOp(ng_op_map, op->name(),
               ConstructNgNode<ng::op::Relu>(op->name() + "_FusedConv2D_Relu",
                                             ng_add));
    } else if (VecStrCmp(fused_ops, {"BiasAdd", "Relu6"})) {
      SaveNgOp(ng_op_map, op->name(), create_relu6(op->name(), ng_add));
    } else {
      SaveNgOp(ng_op_map, op->name(), ng_add);
    }
  } else if (VecStrCmp(fused_ops, {"FusedBatchNorm"}) ||
             VecStrCmp(fused_ops, {"FusedBatchNorm", "Relu"}) ||
             VecStrCmp(fused_ops, {"FusedBatchNorm", "Relu6"})) {
    if (num_args != 4) {
      return errors::InvalidArgument(
          "FusedConv2D with FusedBatchNorm has incompatible num_args");
    }

    shared_ptr<ng::Node> ng_input, ng_filter, ng_conv, ng_scale, ng_offset,
        ng_mean, ng_variance;
    TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_filter,
                                     &ng_scale, &ng_offset, &ng_mean,
                                     &ng_variance));
    TF_RETURN_IF_ERROR(CreateNgConv(ng_input, ng_filter, ng_conv));

    float tf_epsilon;
    TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "epsilon", &tf_epsilon));

    std::shared_ptr<ng::Node> ng_batch_norm =
        ConstructNgNode<ng::op::BatchNormInference>(
            op->name() + "_FusedConv2D_BatchNorm", tf_epsilon, ng_scale,
            ng_offset, ng_conv, ng_mean, ng_variance);

    BatchToTensorflow(op->name(), is_nhwc, ng_batch_norm);

    if (VecStrCmp(fused_ops, {"FusedBatchNorm", "Relu"})) {
      SaveNgOp(ng_op_map, op->name(),
               ConstructNgNode<ng::op::Relu>(
                   op->name() + "_FusedConv2D_BatchNormRelu", ng_batch_norm));
    } else if (VecStrCmp(fused_ops, {"FusedBatchNorm", "Relu6"})) {
      SaveNgOp(ng_op_map, op->name(), create_relu6(op->name(), ng_batch_norm));
    } else {
      SaveNgOp(ng_op_map, op->name(), ng_batch_norm);
    }
  } else {
    return errors::Unimplemented("Unsupported _FusedConv2D " +
                                 absl::StrJoin(fused_ops, ","));
  }
  return Status::OK();
}

static Status TranslateIdentityOp(const Node* op,
                                  const std::vector<const Tensor*>&,
                                  Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_arg;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_arg));
  SaveNgOp(ng_op_map, op->name(), ng_arg);
  return Status::OK();
}

static Status TranslateL2LossOp(const Node* op,
                                const std::vector<const Tensor*>&,
                                Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  auto const_2 = ConstructNgNode<ng::op::Constant>(
      op->name(), ng_input->get_element_type(), ng::Shape{},
      std::vector<std::string>{"2"});

  std::shared_ptr<ng::Node> ng_pow =
      ConstructNgNode<ng::op::Multiply>(op->name(), ng_input, ng_input);

  size_t input_rank = ng_input->get_shape().size();
  ng::AxisSet axes;
  for (size_t i = 0; i < input_rank; ++i) {
    axes.insert(i);
  }

  std::shared_ptr<ng::Node> ng_sum =
      ConstructNgNode<ng::op::Sum>(op->name(), ng_pow, axes);
  std::shared_ptr<ng::Node> ng_l2loss =
      ConstructNgNode<ng::op::Divide>(op->name(), ng_sum, const_2);
  SaveNgOp(ng_op_map, op->name(), ng_l2loss);
  return Status::OK();
}

static Status TranslateLogSoftmaxOp(const Node* op,
                                    const std::vector<const Tensor*>&,
                                    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_inp;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_inp));
  auto inp_shape = ng_inp->get_shape();
  size_t rank = inp_shape.size();
  auto ng_axis = ng::AxisSet{rank - 1};
  // Batch i, class j
  // logsoftmax[i, j] = logits[i, j] - log(sum(exp(logits[i])))
  // Actually implementing: logsoftmax[i, j] = logits[i, j] - max(logits[i]) -
  // log(sum(exp(logits[i] - max(logits[i]))))
  auto ng_max = ConstructNgNode<ng::op::Broadcast>(
      op->name(), ConstructNgNode<ng::op::Max>(op->name(), ng_inp, ng_axis),
      inp_shape, ng_axis);
  auto ng_inp_minus_max =
      ConstructNgNode<ng::op::Subtract>(op->name(), ng_inp, ng_max);
  auto ng_exp = ConstructNgNode<ng::op::Exp>(op->name(), ng_inp_minus_max);
  auto ng_log_sum = ConstructNgNode<ng::op::Log>(
      op->name(), ConstructNgNode<ng::op::Sum>(op->name(), ng_exp, ng_axis));
  auto ng_broadcast = ConstructNgNode<ng::op::Broadcast>(
      op->name(), ng_log_sum, ng_inp->get_shape(), ng_axis);
  auto ng_output = ConstructNgNode<ng::op::Subtract>(
      op->name(), ng_inp_minus_max, ng_broadcast);
  SaveNgOp(ng_op_map, op->name(), ng_output);
  return Status::OK();
}

static Status TranslateSoftplusOp(const Node* op,
                                  const std::vector<const Tensor*>&,
                                  Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_inp;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_inp));
  auto ng_exp = ConstructNgNode<ng::op::Exp>(op->name(), ng_inp);
  auto constant_1 = ConstructNgNode<ng::op::Constant>(
      op->name(), ng_inp->get_element_type(), ng_inp->get_shape(),
      std::vector<std::string>(ng::shape_size(ng_inp->get_shape()), "1"));
  auto ng_output = ConstructNgNode<ng::op::Log>(
      op->name(), ConstructNgNode<ng::op::Add>(op->name(), ng_exp, constant_1));
  SaveNgOp(ng_op_map, op->name(), ng_output);
  return Status::OK();
}

static Status TranslateMatMulOp(const Node* op,
                                const std::vector<const Tensor*>&,
                                Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_lhs, ng_rhs;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_lhs, &ng_rhs));

  // Transpose arguments if requested.
  bool transpose_a = false;
  bool transpose_b = false;

  if (GetNodeAttr(op->attrs(), "transpose_a", &transpose_a) == Status::OK() &&
      transpose_a) {
    ng_lhs = ng::builder::numpy_transpose(ng_lhs, ng::AxisVector{1, 0});
    Builder::SetTracingInfo(op->name(), ng_lhs);
  }
  if (GetNodeAttr(op->attrs(), "transpose_b", &transpose_b) == Status::OK() &&
      transpose_b) {
    ng_rhs = ng::builder::numpy_transpose(ng_rhs, ng::AxisVector{1, 0});
    Builder::SetTracingInfo(op->name(), ng_rhs);
  }

  // The default axis count for nGraph's Dot op is 1, which is just what
  // we need here.
  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<ngraph::op::Dot>(op->name(), ng_lhs, ng_rhs));
  return Status::OK();
}

static Status TranslateMaxPoolOp(const Node* op,
                                 const std::vector<const Tensor*>&,
                                 Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  std::vector<int32> tf_strides;
  std::vector<int32> tf_ksize;
  std::string tf_padding_type;
  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "ksize", &tf_ksize));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument(
        "MaxPool data format is neither NHWC nor NCHW");
  }

  bool is_nhwc = (tf_data_format == "NHWC");

  NGRAPH_VLOG(3) << ng::join(tf_strides);
  NGRAPH_VLOG(3) << ng::join(tf_ksize);
  NGRAPH_VLOG(3) << tf_padding_type;
  NGRAPH_VLOG(3) << tf_data_format;

  ng::Strides ng_strides(2);
  ng::Shape ng_image_shape(2);
  ng::Shape ng_kernel_shape(2);

  BatchedOpParamToNGraph(is_nhwc, tf_strides, ng_strides);
  BatchedOpParamToNGraph(is_nhwc, ng_input->get_shape(), ng_image_shape);
  BatchedOpParamToNGraph(is_nhwc, tf_ksize, ng_kernel_shape);
  BatchToNGraph(op->name(), is_nhwc, ng_input);
  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);
  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  // TODO: change this once nGraph supports negative padding
  // (CoordinateDiff) for MaxPool
  // ng::CoordinateDiff ng_padding_below{0,0};
  // ng::CoordinateDiff ng_padding_above{0,0};
  ng::Shape ng_padding_below{0, 0};
  ng::Shape ng_padding_above{0, 0};

  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_padding_below, ng_padding_above);

  std::shared_ptr<ng::Node> ng_maxpool = ConstructNgNode<ng::op::MaxPool>(
      op->name(), ng_input, ng_kernel_shape, ng_strides, ng_padding_below,
      ng_padding_above);

  BatchToTensorflow(op->name(), is_nhwc, ng_maxpool);

  NGRAPH_VLOG(3) << "maxpool outshape: {" << ng::join(ng_maxpool->get_shape())
                 << "}";

  SaveNgOp(ng_op_map, op->name(), ng_maxpool);
  return Status::OK();
}

static Status TranslateMaxPool3DOp(const Node* op,
                                   const std::vector<const Tensor*>&,
                                   Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  std::vector<int32> tf_strides;
  std::vector<int32> tf_ksize;
  std::string tf_padding_type;
  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "ksize", &tf_ksize));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if (tf_data_format != "NDHWC" && tf_data_format != "NCDHW") {
    return errors::InvalidArgument(
        "MaxPool3D data format is neither NDHWC nor NCDHW");
  }

  bool is_ndhwc = (tf_data_format == "NDHWC");

  NGRAPH_VLOG(3) << ng::join(tf_strides);
  NGRAPH_VLOG(3) << ng::join(tf_ksize);
  NGRAPH_VLOG(3) << tf_padding_type;
  NGRAPH_VLOG(3) << tf_data_format;

  ng::Strides ng_strides(3);
  ng::Shape ng_image_shape(3);
  ng::Shape ng_kernel_shape(3);

  BatchedOpParam3DToNGraph(is_ndhwc, tf_strides, ng_strides);
  BatchedOpParam3DToNGraph(is_ndhwc, ng_input->get_shape(), ng_image_shape);
  BatchedOpParam3DToNGraph(is_ndhwc, tf_ksize, ng_kernel_shape);
  BatchToNGraph3D(op->name(), is_ndhwc, ng_input);
  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);
  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  // TODO: change this once nGraph supports negative padding
  // (CoordinateDiff) for MaxPool
  // ng::CoordinateDiff ng_padding_below{0,0};
  // ng::CoordinateDiff ng_padding_above{0,0};
  ng::Shape ng_padding_below{0, 0, 0};
  ng::Shape ng_padding_above{0, 0, 0};

  Builder::MakePadding3D(tf_padding_type, ng_image_shape, ng_kernel_shape,
                         ng_strides, ng_padding_below, ng_padding_above);

  std::shared_ptr<ng::Node> ng_maxpool = ConstructNgNode<ng::op::MaxPool>(
      op->name(), ng_input, ng_kernel_shape, ng_strides, ng_padding_below,
      ng_padding_above);

  BatchToTensorflow3D(op->name(), is_ndhwc, ng_maxpool);

  NGRAPH_VLOG(3) << "maxpool outshape: {" << ng::join(ng_maxpool->get_shape())
                 << "}";

  SaveNgOp(ng_op_map, op->name(), ng_maxpool);
  return Status::OK();
}

static Status TranslateMaxPoolGradOp(const Node* op,
                                     const std::vector<const Tensor*>&,
                                     Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_grad, ng_fwd;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, &ng_input, &ng_fwd, &ng_grad));

  std::vector<int32> tf_strides;
  std::vector<int32> tf_ksize;
  std::string tf_padding_type;
  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "ksize", &tf_ksize));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));
  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument(
        "MaxPoolGrad data format is neither NHWC nor NCHW");
  }

  bool is_nhwc = (tf_data_format == "NHWC");
  NGRAPH_VLOG(3) << ng::join(tf_strides);
  NGRAPH_VLOG(3) << ng::join(tf_ksize);
  NGRAPH_VLOG(3) << tf_padding_type;
  NGRAPH_VLOG(3) << tf_data_format;

  ng::Strides ng_strides(2);
  ng::Shape ng_image_shape(2);
  ng::Shape ng_kernel_shape(2);

  BatchedOpParamToNGraph(is_nhwc, ng_input->get_shape(), ng_image_shape);
  BatchedOpParamToNGraph(is_nhwc, tf_strides, ng_strides);
  BatchedOpParamToNGraph(is_nhwc, tf_ksize, ng_kernel_shape);
  BatchToNGraph(op->name(), is_nhwc, ng_input);
  BatchToNGraph(op->name(), is_nhwc, ng_grad);
  BatchToNGraph(op->name(), is_nhwc, ng_fwd);

  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);
  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  ng::Shape ng_padding_below{0, 0};
  ng::Shape ng_padding_above{0, 0};

  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_padding_below, ng_padding_above);

  std::shared_ptr<ng::Node> ng_maxpool_backprop =
      ConstructNgNode<ng::op::MaxPoolBackprop>(
          op->name(), ng_input, ng_grad, ng_fwd, ng_kernel_shape, ng_strides,
          ng_padding_below, ng_padding_above);
  BatchToTensorflow(op->name(), is_nhwc, ng_maxpool_backprop);
  NGRAPH_VLOG(3) << "maxpoolbackprop outshape: {"
                 << ng::join(ng_maxpool_backprop->get_shape()) << "}";
  SaveNgOp(ng_op_map, op->name(), ng_maxpool_backprop);
  return Status::OK();
}

static Status TranslateNonMaxSuppressionV4Op(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_boxes, ng_scores;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_boxes, &ng_scores,
                                   nullptr, nullptr, nullptr));

  std::vector<int> max_output_size;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(op, 2, static_input_map, &max_output_size));
  std::vector<float> iou_threshold;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(op, 3, static_input_map, &iou_threshold));

  std::vector<float> score_threshold;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(op, 4, static_input_map, &score_threshold));

  bool pad_to_max_output_size;
  if (GetNodeAttr(op->attrs(), "pad_to_max_output_size",
                  &pad_to_max_output_size) != Status::OK()) {
    pad_to_max_output_size = false;
  }
  // max_output_size must be scalar
  if (max_output_size.size() != 1) {
    return errors::InvalidArgument(
        "NonMaxSuppressionV4 Op: max_output_size of nms must be scalar ",
        max_output_size.size());
  }
  // iou_threshold must be scalar
  if (iou_threshold.size() != 1) {
    return errors::InvalidArgument(
        "NonMaxSuppressionV4 Op: iou_threshold of nms must be scalar ",
        iou_threshold.size());
  }

  // score_threshold must be scalar
  if (score_threshold.size() != 1) {
    return errors::InvalidArgument(
        "NonMaxSuppressionV4 Op: score_threshold of nms must be scalar ",
        score_threshold.size());
  }

  std::string backend_name;
  TF_RETURN_IF_ERROR(ngraph_bridge::GetNodeBackend(op, &backend_name));

  auto config_map = BackendManager::GetBackendAttributeValues(backend_name);
  if (config_map.at("ngraph_backend") != "NNPI") {
    return errors::Internal("In translating NonMaxSuppressionV4 op ",
                            op->name(), " found requested backend ",
                            backend_name, " which is unsupported");
  }

  ng::runtime::Backend* backend = BackendManager::GetBackend(backend_name);

  shared_ptr<ng::Node> ng_nmsv4 = backend->get_backend_op(
      "NonMaxSuppressionV4", &ng_boxes, &ng_scores,
      (size_t)(max_output_size[0]), (float)(iou_threshold[0]),
      (float)score_threshold[0], (bool)pad_to_max_output_size);
  if (ng_nmsv4 == nullptr) {
    return errors::Internal("In translating NonMaxSuppressionV4 op ",
                            op->name(),
                            " backend could not return valid ngraph node");
  }
  Builder::SetTracingInfo(op->name(), ng_nmsv4);
  shared_ptr<ngraph::Node> ng_selected_indices =
      ConstructNgNode<ngraph::op::GetOutputElement>(op->name(), ng_nmsv4, 0);
  shared_ptr<ngraph::Node> ng_valid_output =
      ConstructNgNode<ngraph::op::GetOutputElement>(op->name(), ng_nmsv4, 1);

  SaveNgOp(ng_op_map, op->name(), ng_selected_indices);
  SaveNgOp(ng_op_map, op->name(), ng_valid_output);

  return Status::OK();
}

static Status TranslateReduceOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map,
    std::function<std::shared_ptr<ng::Node>(std::shared_ptr<ng::Node>,
                                            ng::AxisSet)>
        create_ng_node) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, &ng_input));
  bool tf_keep_dims;
  if (GetNodeAttr(op->attrs(), "keep_dims", &tf_keep_dims) != Status::OK()) {
    tf_keep_dims = false;
  }

  std::vector<int64> axes;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &axes));

  ng::Shape input_shape = ng_input->get_shape();
  size_t input_rank = input_shape.size();

  TF_RETURN_IF_ERROR(CheckAxisDimInRange(axes, input_rank));

  std::vector<size_t> ng_reduction_axes_vect(axes.size());
  std::transform(
      axes.begin(), axes.end(), ng_reduction_axes_vect.begin(),
      [input_rank](int idx) { return idx + (idx < 0 ? (int)input_rank : 0); });
  ng::AxisSet ng_reduction_axes(ng_reduction_axes_vect);

  std::shared_ptr<ng::Node> ng_node =
      create_ng_node(ng_input, ng_reduction_axes);
  Builder::SetTracingInfo(op->name(), ng_node);

  // If keep_dims is specified we need to reshape to put back the reduced
  // axes, with length 1.
  if (tf_keep_dims) {
    ng::Shape ng_result_shape_with_keep(input_rank);

    for (size_t i = 0; i < input_rank; i++) {
      ng_result_shape_with_keep[i] =
          ng_reduction_axes.count(i) == 0 ? input_shape[i] : 1;
    }

    ng::AxisVector ng_axis_order(ng_node->get_shape().size());
    std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);

    ng_node = ConstructNgNode<ng::op::Reshape>(
        op->name(), ng_node, ng_axis_order, ng_result_shape_with_keep);
  }

  SaveNgOp(ng_op_map, op->name(), ng_node);
  return Status::OK();
}

static Status TranslateMeanOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  string op_name = op->name();
  return TranslateReduceOp(op, static_input_map, ng_op_map,
                           [&op_name](std::shared_ptr<ng::Node> ng_input,
                                      ng::AxisSet ng_reduction_axes) {
                             auto mean_node =
                                 ng::builder::mean(ng_input, ng_reduction_axes);
                             Builder::SetTracingInfo(op_name, mean_node);
                             return mean_node;
                           });
}

template <typename T>
static Status TranslateDirectReduceOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  // ensure its Any, All, Min, Max, Sum or Product
  if (!(std::is_same<T, ng::op::Sum>::value ||
        std::is_same<T, ng::op::Product>::value ||
        std::is_same<T, ng::op::Max>::value ||
        std::is_same<T, ng::op::Min>::value ||
        std::is_base_of<ngraph::op::util::LogicalReduction, T>::value)) {
    return errors::InvalidArgument(
        "Expected node to be Any, All, Min, Max, Sum or Product type");
  }
  return TranslateReduceOp(
      op, static_input_map, ng_op_map,
      [&op](std::shared_ptr<ng::Node> ng_input, ng::AxisSet ng_reduction_axes) {
        return ConstructNgNode<T>(op->name(), ng_input, ng_reduction_axes);
      });
}

static Status TranslateOneHotOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_features, ng_on, ng_off;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, &ng_features, nullptr, &ng_on, &ng_off));

  auto ng_features_shape = ng_features->get_shape();
  auto ng_features_rank = ng_features_shape.size();

  std::vector<int> depth;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &depth));

  // Depth must be scalar
  if (depth.size() != 1) {
    return errors::InvalidArgument(
        "OneHot Op: depth of one hot dimension must be scalar ", depth.size());
  }

  int one_hot_axis;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "axis", &one_hot_axis));

  ng::Shape output_shape(ng_features_shape);
  auto pos = output_shape.begin();
  if (one_hot_axis == -1) {
    one_hot_axis = ng_features_rank;
    pos = output_shape.end();
  } else {
    pos = output_shape.begin() + one_hot_axis;
  }
  output_shape.insert(pos, depth[0]);

  auto ng_onehot_labels = ConstructNgNode<ng::op::OneHot>(
      op->name(), ng_features, output_shape, one_hot_axis);

  shared_ptr<ng::Node> ng_onehot_bool = ConstructNgNode<ng::op::Convert>(
      op->name(), ng_onehot_labels, ng::element::boolean);

  // broadcast to make all tensors same shape, as required by ngraph select op
  std::tie(ng_onehot_bool, ng_on) =
      Builder::PerformNgBroadcast(op->name(), ng_onehot_bool, ng_on);
  std::tie(ng_onehot_bool, ng_off) =
      Builder::PerformNgBroadcast(op->name(), ng_onehot_bool, ng_off);

  auto ng_onehot = ConstructNgNode<ng::op::Select>(op->name(), ng_onehot_bool,
                                                   ng_on, ng_off);

  SaveNgOp(ng_op_map, op->name(), ng_onehot);
  return Status::OK();
}

static Status TranslatePackOp(const Node* op, const std::vector<const Tensor*>&,
                              Builder::OpMap& ng_op_map) {
  TF_RETURN_IF_ERROR(ValidateInputCountMin(op, 1));

  ng::NodeVector ng_concat_inputs;

  for (tensorflow::int32 i = 0; i < op->num_inputs(); ++i) {
    shared_ptr<ng::Node> ng_input;
    TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, i, &ng_input));
    ng_concat_inputs.push_back(ng_input);
  }

  int32 tf_axis;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "axis", &tf_axis));
  size_t input_rank = ng_concat_inputs[0]->get_shape().size();

  auto concat_axis = tf_axis;
  if (concat_axis == -1) {
    concat_axis = input_rank;
  }

  ng::Shape input_shape = ng_concat_inputs[0]->get_shape();
  ng::Shape output_shape(input_rank + 1);

  // if inputs shape is (2, 3, 4), and axis is 1, then we want
  // to create output_shape (2, num_inputs, 3, 4)
  for (size_t i = 0; i < input_rank; ++i) {
    output_shape[((int)i < concat_axis) ? i : i + 1] = input_shape[i];
  }
  output_shape[concat_axis] = op->num_inputs();

  ng::AxisVector ng_axis_order(input_rank);
  std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);

  if ((size_t)concat_axis == input_rank) {
    // need to add extra dimension before we concatenate
    // along it
    ng::Shape extended_shape = input_shape;
    extended_shape.push_back(1);
    for (size_t i = 0; i < ng_concat_inputs.size(); ++i) {
      ng_concat_inputs[i] = ConstructNgNode<ng::op::Reshape>(
          op->name(), ng_concat_inputs[i], ng_axis_order, extended_shape);
    }
    ng_axis_order.push_back(input_rank);
  }

  auto concat = ConstructNgNode<ng::op::Concat>(op->name(), ng_concat_inputs,
                                                concat_axis);
  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<ng::op::Reshape>(op->name(), concat, ng_axis_order,
                                            output_shape));
  return Status::OK();
}

static Status TranslatePadOp(const Node* op,
                             const std::vector<const Tensor*>& static_input_map,
                             Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_paddings_op;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_paddings_op));

  std::vector<int64> paddings;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &paddings));

  NGRAPH_VLOG(3) << "{" << ng::join(paddings) << "}";

  if (paddings.size() % 2 != 0) {
    return errors::InvalidArgument(
        "Constant node for paddings does not have an even number of "
        "elements");
  }

  ng::CoordinateDiff padding_below(paddings.size() / 2);
  ng::CoordinateDiff padding_above(paddings.size() / 2);
  ng::Shape padding_interior(paddings.size() / 2);
  auto pad_mode = ng::op::PadMode::CONSTANT;

  for (size_t i = 0; i < paddings.size() / 2; i++) {
    padding_below[i] = paddings[2 * i];
    padding_above[i] = paddings[2 * i + 1];
    padding_interior[i] = 0;
  }

  NGRAPH_VLOG(3) << "{" << ng::join(padding_below) << "}";
  NGRAPH_VLOG(3) << "{" << ng::join(padding_above) << "}";

  // For PadV1 it seems the value is always zero.
  auto pad_val_op = ConstructNgNode<ng::op::Constant>(
      op->name(), ng_input->get_element_type(), ng::Shape{},
      std::vector<std::string>{"0"});
  auto pad_op = ConstructNgNode<ng::op::Pad>(
      op->name(), ng_input, pad_val_op, padding_below, padding_above, pad_mode);

  SaveNgOp(ng_op_map, op->name(), pad_op);
  return Status::OK();
}

static Status TranslateRankOp(const Node* op, const std::vector<const Tensor*>&,
                              Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;

  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  ng::Shape input_shape = ng_input->get_shape();
  auto input_rank = static_cast<int>(input_shape.size());

  auto ng_rank = ConstructNgNode<ng::op::Constant>(
      op->name(), ng::element::i32, ng::Shape(),
      std::vector<int>({input_rank}));

  SaveNgOp(ng_op_map, op->name(), ng_rank);
  return Status::OK();
}

static Status TranslateReciprocalOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateUnaryOp(
      op, static_input_map, ng_op_map, [&op](std::shared_ptr<ng::Node> n) {
        // Create a constant tensor populated with the value -1.
        // (1/x = x^(-1))
        auto et = n->get_element_type();
        auto shape = n->get_shape();
        std::vector<std::string> constant_values(ng::shape_size(shape), "-1");
        auto ng_exponent = ConstructNgNode<ng::op::Constant>(
            op->name(), et, shape, constant_values);

        // Raise each element of the input to the power -1.
        return ConstructNgNode<ng::op::Power>(op->name(), n, ng_exponent);
      });
}

template <typename T>
Status QuantizeAndDequantizeV2Helper(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    const bool& range_given, const bool& signed_input, const int& num_bits,
    float* scale_out) {
  // TODO: currently handling only float, generalize later?
  T min_range = 0, max_range = 0;
  if (range_given) {
    std::vector<T> input_min, input_max;
    TF_RETURN_IF_ERROR(
        GetStaticInputVector(op, 1, static_input_map, &input_min));
    TF_RETURN_IF_ERROR(
        GetStaticInputVector(op, 2, static_input_map, &input_max));
    if (input_min.size() != 1) {
      return errors::InvalidArgument(
          "QuantizeAndDequantizeV2 Op: input_min must be scalar. Got a "
          "vector "
          "of size, ",
          input_min.size());
    }
    if (input_max.size() != 1) {
      return errors::InvalidArgument(
          "QuantizeAndDequantizeV2 Op: input_max must be scalar. Got a "
          "vector "
          "of size, ",
          input_max.size());
    }
    min_range = input_min[0];
    max_range = input_max[0];
    if (min_range > max_range) {
      return errors::InvalidArgument(
          "Expected QuantizeAndDequantizeV2's input_min <= input_max but "
          "got, "
          "input_min = ",
          min_range, " and input_max = ", max_range);
    }
    // m = max(abs(input_min), abs(input_max));
  } else {
    // m = max(abs(min_elem(input)), abs(max_elem(input)));
    // TODO implement this.
    // Note to implement this we need:
    // min = ng_min(inp_tensor); max = mg_max(inp_tensor).
    // which means, unless we support pattern matching that accepts the ng min
    // and max nodes, we have to declare inp_data tensor to be static
  }
  const int64 min_quantized = signed_input ? -(1ULL << (num_bits - 1)) : 0;
  const int64 max_quantized = min_quantized + ((1ULL << num_bits) - 1);
  const T scale_from_min_side = (min_quantized * min_range > 0)
                                    ? min_quantized / min_range
                                    : std::numeric_limits<T>::max();
  const T scale_from_max_side = (max_quantized * max_range > 0)
                                    ? max_quantized / max_range
                                    : std::numeric_limits<T>::max();
  T inverse_scale;
  if (scale_from_min_side < scale_from_max_side && min_quantized != 0) {
    // min_quantized != 0 is not really necessary but klocwork complains
    // T scale = scale_from_min_side;
    inverse_scale = min_range / min_quantized;
    // max_range = max_quantized * inverse_scale;
  } else {
    // T scale = scale_from_max_side;
    inverse_scale = max_range / max_quantized;
    // min_range = min_quantized * inverse_scale;
  }
  *scale_out = inverse_scale;

  return Status::OK();
}

static Status TranslateQuantizeAndDequantizeV2Op(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, nullptr, nullptr));
  bool range_given;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "range_given", &range_given));

  bool signed_input;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "signed_input", &signed_input));

  int num_bits;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "num_bits", &num_bits));

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "T", &dtype));
  // T: float, double....not supported: bfloat16, half
  float scale;
  ng::element::Type ng_r_et;
  switch (dtype) {
    case DT_FLOAT:
      TF_RETURN_IF_ERROR(QuantizeAndDequantizeV2Helper<float>(
          op, static_input_map, range_given, signed_input, num_bits, &scale));
      ng_r_et = ng::element::f32;
      break;
    case DT_DOUBLE:
      TF_RETURN_IF_ERROR(QuantizeAndDequantizeV2Helper<double>(
          op, static_input_map, range_given, signed_input, num_bits, &scale));
      ng_r_et = ng::element::f64;
      break;
    default:
      return errors::InvalidArgument(
          "Expected QuantizeAndDequantizeV2's datatype to be of DT_FLOAT or "
          "DT_DOUBLE but got ",
          DataTypeString(dtype));
  }
  // The quantized data type
  ng::element::Type ng_q_et;
  switch (num_bits) {
    case 8:
      ng_q_et = signed_input ? ng::element::i8 : ng::element::u8;
      break;
    default:
      return errors::InvalidArgument(
          "Expected QuantizeAndDequantizeV2's num_bits to be 8, but got ",
          num_bits);
  }
  auto ng_scale = ConstructNgNode<ng::op::Constant>(
      op->name(), ng_r_et, ng::Shape(), std::vector<float>({scale}));
  auto ng_offset = ConstructNgNode<ng::op::Constant>(
      op->name(), ng_q_et, ng::Shape(), std::vector<int>({0}));
  ng::op::Quantize::RoundMode ng_round_mode =
      ng::op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_INFINITY;
  auto ng_quant = ConstructNgNode<ng::op::Quantize>(
      op->name(), ng_input, ng_scale, ng_offset, ng_q_et, ng::AxisSet(),
      ng_round_mode);
  SaveNgOp(ng_op_map, op->name(), ConstructNgNode<ng::op::Dequantize>(
                                      op->name(), ng_quant, ng_scale, ng_offset,
                                      ng_r_et, ng::AxisSet()));

  // TODO: what of clamping?
  return Status::OK();
}

static Status TranslateQuantizedAvgPoolOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateQuantizedPoolOp(op, static_input_map, ng_op_map,
                                  "QuantizedAvgPool");
}

// Helper function to translate QuantizedConcat and QuantizedConcatV2
static Status TranslateQuantizedConcatOpHelper(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map, std::string op_name) {
  int axis_index;         // index for concat_axis input
  int value_start_index;  // start index for N tensor inputs
  auto num_of_tensors_to_concat = (op->num_inputs() - 1) / 3;

  if (op_name == "QuantizedConcat") {
    axis_index = 0;
    value_start_index = 1;
  } else if (op_name == "QuantizedConcatV2") {
    axis_index = num_of_tensors_to_concat;
    value_start_index = 0;
  } else {
    return errors::InvalidArgument(
        "This helper function is only used for QuantizedConcat and "
        "QuantizedConcatV2 ops");
  }

  auto collect_nodes = [&op, &ng_op_map](int start, int end,
                                         ng::NodeVector* p_ng_args) {
    for (int i = start; i < end; i++) {
      shared_ptr<ng::Node> ng_arg;
      TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, i, &ng_arg));
      (*p_ng_args).push_back(ng_arg);
    }
    return Status::OK();
  };

  // Collect the N input tensors to concat with
  ng::NodeVector ng_args;
  if (op_name == "QuantizedConcat") {
    TF_RETURN_IF_ERROR(collect_nodes(value_start_index,
                                     num_of_tensors_to_concat + 1, &ng_args));
  } else if (op_name == "QuantizedConcatV2") {
    TF_RETURN_IF_ERROR(
        collect_nodes(value_start_index, num_of_tensors_to_concat, &ng_args));
  }

  // Get the input concat_axis
  std::vector<int64> tf_concat_axis_vec;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, axis_index, static_input_map,
                                          &tf_concat_axis_vec));

  // QuantizedConcat doesn't have negative concat_axis
  int64 concat_axis = tf_concat_axis_vec[0];

  // Get input_mins and input_maxs
  std::vector<float> all_mins(num_of_tensors_to_concat),
      all_maxs(num_of_tensors_to_concat);

  // Construct input parameters to ScaledQuantizedConcat op
  ng::NodeVector ng_all_mins, ng_all_maxs;
  std::vector<float> min_tmp, max_tmp;

  // Collect the N input mins and input maxes
  for (int idx = 0; idx < num_of_tensors_to_concat; idx++) {
    TF_RETURN_IF_ERROR(GetStaticInputVector(
        op, num_of_tensors_to_concat + 1 + idx, static_input_map, &min_tmp));
    TF_RETURN_IF_ERROR(
        GetStaticInputVector(op, 2 * num_of_tensors_to_concat + 1 + idx,
                             static_input_map, &max_tmp));

    all_mins[idx] = min_tmp[0];
    all_maxs[idx] = max_tmp[0];

    auto min_node = ConstructNgNode<ng::op::Constant>(
        op->name(), ng::element::f32, ng::Shape{}, min_tmp);
    auto max_node = ConstructNgNode<ng::op::Constant>(
        op->name(), ng::element::f32, ng::Shape{}, max_tmp);

    ng_all_mins.push_back(ConstructNgNode<ngraph::op::Reshape>(
        op->name(), min_node, ngraph::AxisVector{}, ngraph::Shape{1}));
    ng_all_maxs.push_back(ConstructNgNode<ngraph::op::Reshape>(
        op->name(), max_node, ngraph::AxisVector{}, ngraph::Shape{1}));
  }

  // return the min among the input_mins, and the max among the input_maxs
  // TODO: TF has a different way of determine the output_min and output_max
  // TF reference:
  // https://github.com/tensorflow/tensorflow/blob/86950c2c440be956a9fcb3a25868a1df15444467/tensorflow/core/kernels/quantized_concat_op.cc#L78
  std::vector<float> min_of_mins(
      1, *std::min_element(all_mins.begin(), all_mins.end()));
  std::vector<float> max_of_maxs(
      1, *std::max_element(all_maxs.begin(), all_maxs.end()));

  // construct output_min and output_max
  shared_ptr<ng::Node> ng_min_of_mins = ConstructNgNode<ng::op::Constant>(
      op->name(), ng::element::f32, ng::Shape{}, min_of_mins);
  shared_ptr<ng::Node> ng_max_of_maxs = ConstructNgNode<ng::op::Constant>(
      op->name(), ng::element::f32, ng::Shape{}, max_of_maxs);

  auto ng_qconcat = ng::builder::ScaledQuantizedConcat(
      ng_args, size_t(concat_axis), ng_all_mins, ng_all_maxs);
  Builder::SetTracingInfo(op->name(), ng_qconcat);

  SaveNgOp(ng_op_map, op->name(), ng_qconcat);
  SaveNgOp(ng_op_map, op->name(), ng_min_of_mins);
  SaveNgOp(ng_op_map, op->name(), ng_max_of_maxs);
  return Status::OK();
}

static Status TranslateQuantizedConcatOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateQuantizedConcatOpHelper(op, static_input_map, ng_op_map,
                                          "QuantizedConcat");
}

static Status TranslateQuantizedConcatV2Op(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateQuantizedConcatOpHelper(op, static_input_map, ng_op_map,
                                          "QuantizedConcatV2");
}

static Status TranslateQuantizedConv(
    const Node* op, Builder::OpMap& ng_op_map,
    std::function<std::shared_ptr<ng::Node>(
        std::vector<std::shared_ptr<ng::Node>>, ng::Strides, ng::Strides,
        ng::CoordinateDiff, ng::CoordinateDiff, ng::Strides)>
        create_quantized_conv_node) {
  size_t num_tf_op_inputs = op->num_inputs();
  size_t num_node_inputs = num_tf_op_inputs;
  std::vector<std::shared_ptr<ng::Node>> node_inps(num_node_inputs);
  for (size_t inp_idx = 0; inp_idx < num_tf_op_inputs; inp_idx++) {
    TF_RETURN_IF_ERROR(
        GetInputNode(ng_op_map, op, inp_idx, &(node_inps[inp_idx])));
  }

  std::vector<int32> tf_strides;
  std::vector<int32> tf_dilations;
  std::string tf_padding_type;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "dilations", &tf_dilations));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
  bool is_nhwc = true;  // TODO: Assuming this data format for now
  ng::Strides ng_strides(2);
  ng::Strides ng_dilations(2);
  ng::Strides ng_data_dilations({1, 1});
  ng::Shape ng_image_shape(2);
  ng::Shape ng_kernel_shape(2);
  BatchedOpParamToNGraph(is_nhwc, tf_strides, ng_strides);
  BatchedOpParamToNGraph(is_nhwc, node_inps[0]->get_shape(), ng_image_shape);
  BatchedOpParamToNGraph(is_nhwc, tf_dilations, ng_dilations);
  // Generally, the mapping is: 0->input, 1->filter, 2->bias, 3->sum input
  BatchToNGraph(op->name(), is_nhwc, node_inps[0]);
  // QconvBiasAdd variants
  if (num_node_inputs == 12) {
    BatchToNGraph(op->name(), is_nhwc, node_inps[9]);
  }
  auto& ng_filter_shape = node_inps[1]->get_shape();
  ng_kernel_shape[0] = ng_filter_shape[0];
  ng_kernel_shape[1] = ng_filter_shape[1];
  Reshape<3, 2, 0, 1>(node_inps[1]);
  Builder::SetTracingInfo(op->name(), node_inps[1]);
  ng::CoordinateDiff ng_padding_below{0, 0};
  ng::CoordinateDiff ng_padding_above{0, 0};
  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_dilations, ng_padding_below,
                       ng_padding_above);

  // It is expected by ScaledQuantizedConvolutionBias (and other builder
  // functions) that the min max inputs be constant nodes
  // Hence declaring them static, reading their values and converting to
  // constant nodes
  std::shared_ptr<ng::Node> ng_quant_conv_bias = create_quantized_conv_node(
      node_inps, ng_strides, ng_dilations, ng_padding_below, ng_padding_above,
      ng_data_dilations);
  Builder::SetTracingInfo(op->name(), ng_quant_conv_bias);

  BatchToTensorflow(op->name(), is_nhwc, ng_quant_conv_bias);
  SaveNgOp(ng_op_map, op->name(), ng_quant_conv_bias);
  // QconvBiasAdd variants have summand and its min/max as the last input
  // nodes
  auto adjust_idx = num_node_inputs == 12 ? 3 : 0;
  // Forward the min_freezed_output input to output min
  SaveNgOp(ng_op_map, op->name(), node_inps[num_node_inputs - 2 - adjust_idx]);
  // Forward the max_freezed_output input to output max
  SaveNgOp(ng_op_map, op->name(), node_inps[num_node_inputs - 1 - adjust_idx]);
  return Status::OK();
}

template <bool IsRelu>
static Status TranslateQuantizedConv2DWithBiasMaybeReluAndRequantizeOp(
    const Node* op, const std::vector<const Tensor*>&,
    Builder::OpMap& ng_op_map) {
  string op_name = op->name();
  auto create_quantized_conv_node = [&op_name](
      std::vector<std::shared_ptr<ng::Node>> node_inps, ng::Strides ng_strides,
      ng::Strides ng_dilations, ng::CoordinateDiff ng_padding_below,
      ng::CoordinateDiff ng_padding_above, ng::Strides ng_data_dilations) {
    auto ng_node = ng::builder::ScaledQuantizedConvolutionBias(
        node_inps[0], node_inps[1], node_inps[2], ng_strides, ng_dilations,
        ng_padding_below, ng_padding_above, ng_data_dilations, node_inps[3],
        node_inps[4], node_inps[5], node_inps[6], node_inps[7], node_inps[8],
        IsRelu);
    Builder::SetTracingInfo(op_name, ng_node);
    return ng_node;
  };
  return TranslateQuantizedConv(op, ng_op_map, create_quantized_conv_node);
}

static Status TranslateQuantizedConv2DWithBiasSumAndReluAndRequantizeOp(
    const Node* op, const std::vector<const Tensor*>&,
    Builder::OpMap& ng_op_map) {
  string op_name = op->name();
  auto create_quantized_conv_node = [&op_name](
      std::vector<std::shared_ptr<ng::Node>> node_inps, ng::Strides ng_strides,
      ng::Strides ng_dilations, ng::CoordinateDiff ng_padding_below,
      ng::CoordinateDiff ng_padding_above, ng::Strides ng_data_dilations) {
    auto ng_node = ng::builder::ScaledQuantizedConvolutionBiasAdd(
        node_inps[0], node_inps[1], node_inps[2], node_inps[9], ng_strides,
        ng_dilations, ng_padding_below, ng_padding_above, ng_data_dilations,
        node_inps[3], node_inps[4], node_inps[5], node_inps[6], node_inps[7],
        node_inps[8], node_inps[10], node_inps[11], true);
    Builder::SetTracingInfo(op_name, ng_node);
    return ng_node;
  };
  return TranslateQuantizedConv(op, ng_op_map, create_quantized_conv_node);
}

static Status TranslateQuantizedConv2DWithBiasSignedSumAndReluAndRequantizeOp(
    const Node* op, const std::vector<const Tensor*>&,
    Builder::OpMap& ng_op_map) {
  string op_name = op->name();
  auto create_quantized_conv_node = [&op_name](
      std::vector<std::shared_ptr<ng::Node>> node_inps, ng::Strides ng_strides,
      ng::Strides ng_dilations, ng::CoordinateDiff ng_padding_below,
      ng::CoordinateDiff ng_padding_above, ng::Strides ng_data_dilations) {
    auto ng_node = ng::builder::ScaledQuantizedConvolutionBiasSignedAdd(
        node_inps[0], node_inps[1], node_inps[2], node_inps[9], ng_strides,
        ng_dilations, ng_padding_below, ng_padding_above, ng_data_dilations,
        node_inps[3], node_inps[4], node_inps[5], node_inps[6], node_inps[7],
        node_inps[8], node_inps[10], node_inps[11], true);
    Builder::SetTracingInfo(op_name, ng_node);
    return ng_node;
  };
  return TranslateQuantizedConv(op, ng_op_map, create_quantized_conv_node);
}

static Status TranslateQuantizedMaxPoolOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateQuantizedPoolOp(op, static_input_map, ng_op_map,
                                  "QuantizedMaxPool");
}

static Status TranslateQuantizeV2Op(const Node* op,
                                    const std::vector<const Tensor*>&,
                                    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_min, ng_max;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_min, &ng_max));

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "T", &dtype));

  ng::element::Type ng_et;
  TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(dtype, &ng_et));

  // TODO: Only RoundMode = ROUND_NEAREST_TOWARD_EVEN is supported, for now.
  // Support other modes later
  ng::op::Quantize::RoundMode ng_round_mode =
      ng::op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

  auto ng_node = ng::builder::ScaledQuantize(ng_input, ng_min, ng_max, ng_et,
                                             ng::AxisSet(), ng_round_mode);
  Builder::SetTracingInfo(op->name(), ng_node);
  SaveNgOp(ng_op_map, op->name(), ng_node);
  SaveNgOp(ng_op_map, op->name(), ng_min);
  SaveNgOp(ng_op_map, op->name(), ng_max);

  return Status::OK();
}

static Status TranslateDequantizeOp(const Node* op,
                                    const std::vector<const Tensor*>&,
                                    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_min, ng_max;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_min, &ng_max));

  // TF only dequantizes to fp32
  auto ng_node = ng::builder::ScaledDequantize(ng_input, ng_min, ng_max,
                                               ng::element::f32, ng::AxisSet());
  Builder::SetTracingInfo(op->name(), ng_node);
  SaveNgOp(ng_op_map, op->name(), ng_node);
  return Status::OK();
}

static Status TranslateRelu6Op(const Node* op,
                               const std::vector<const Tensor*>&,
                               Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  auto constant_6 = ConstructNgNode<ng::op::Constant>(
      op->name(), ng_input->get_element_type(), ng_input->get_shape(),
      std::vector<std::string>(ng::shape_size(ng_input->get_shape()), "6"));
  auto relu6_op = ConstructNgNode<ng::op::Minimum>(
      op->name(), ConstructNgNode<ng::op::Relu>(op->name(), ng_input),
      constant_6);

  SaveNgOp(ng_op_map, op->name(), relu6_op);
  return Status::OK();
}

static Status TranslateReluGradOp(const Node* op,
                                  const std::vector<const Tensor*>&,
                                  Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_arg, ng_delta;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_delta, &ng_arg));

  auto ng_relu_grad =
      ConstructNgNode<ng::op::ReluBackprop>(op->name(), ng_arg, ng_delta);
  SaveNgOp(ng_op_map, op->name(), ng_relu_grad);
  return Status::OK();
}

static Status TranslateReshapeOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_shape_op;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_shape_op));

  NGRAPH_VLOG(3) << "Input shape: " << ng::join(ng_input->get_shape());

  std::vector<int64> shape;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &shape));

  NGRAPH_VLOG(3) << "Requested result shape: " << ng::join(shape);

  size_t output_rank = shape.size();
  size_t num_input_elements = ng::shape_size(ng_input->get_shape());

  //
  // If there is a single "-1" in the result shape, we have to auto-infer
  // the length of that dimension.
  //
  size_t inferred_pos;
  size_t product_of_rest = 1;
  bool seen_inferred = false;
  for (size_t i = 0; i < output_rank; i++) {
    if (shape[i] == -1) {
      if (seen_inferred) {
        return errors::InvalidArgument(
            "Multiple -1 dimensions in result shape");
      }
      inferred_pos = i;
      seen_inferred = true;
    } else {
      product_of_rest *= shape[i];
    }
  }

  if (seen_inferred) {
    if (num_input_elements % product_of_rest != 0) {
      NGRAPH_VLOG(3) << "{" << ng::join(ng_input->get_shape()) << "}";
      NGRAPH_VLOG(3) << "{" << ng::join(shape) << "}";
      return errors::InvalidArgument(
          "Product of known dimensions (", product_of_rest,
          ") does not evenly divide the number of input elements (",
          num_input_elements, ")");
    }
    shape[inferred_pos] = num_input_elements / product_of_rest;
  }

  //
  // Convert the values from the constant into an nGraph::Shape, and
  // construct the axis order while we are at it.
  //
  ng::Shape ng_shape(output_rank);

  for (size_t i = 0; i < output_rank; i++) {
    ng_shape[i] = shape[i];
  }

  ng::AxisVector ng_axis_order(ng_input->get_shape().size());
  std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);

  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<ng::op::Reshape>(op->name(), ng_input, ng_axis_order,
                                            ng_shape));
  return Status::OK();
}

static Status TranslateRsqrtOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateUnaryOp(
      op, static_input_map, ng_op_map, [&op](std::shared_ptr<ng::Node> n) {
        // Create a constant tensor populated with the value -1/2.
        // (1/sqrt(x) = x^(-1/2))
        auto et = n->get_element_type();
        auto shape = n->get_shape();
        std::vector<std::string> constant_values(ng::shape_size(shape), "-0.5");
        auto ng_exponent = ConstructNgNode<ng::op::Constant>(
            op->name(), et, shape, constant_values);

        // Raise each element of the input to the power -0.5.
        return ConstructNgNode<ng::op::Power>(op->name(), n, ng_exponent);
      });
}

static Status TranslateScatterNdOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_indices;
  shared_ptr<ng::Node> ng_updates;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, &ng_indices, &ng_updates, nullptr));

  std::vector<int> ng_shape;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 2, static_input_map, &ng_shape));
  // Copy the int vector to a size_t vector, because that is what ng::Shape
  // accepts
  std::vector<size_t> ng_shape_size_t(ng_shape.begin(), ng_shape.end());

  // Create a tensor and populate the tensor with "0" to Add to ScatterNd
  auto et = ng_updates->get_element_type();
  std::vector<std::string> constant_values(ng::shape_size(ng_shape_size_t),
                                           "0");
  auto ng_inputs = ConstructNgNode<ng::op::Constant>(
      op->name(), et, ng::Shape(ng_shape_size_t), constant_values);

  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<ng::op::ScatterNDAdd>(op->name(), ng_inputs,
                                                 ng_indices, ng_updates));

  return Status::OK();
}

static Status TranslateRsqrtGradOp(const Node* op,
                                   const std::vector<const Tensor*>&,
                                   Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  shared_ptr<ng::Node> ng_delta;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_delta));

  //`grad = dy * -0.5 * y^3`, where `y = rsqrt(x)`, and `dy`
  // Create a constant tensor populated with the value 3.
  auto et = ng_input->get_element_type();
  auto shape = ng_input->get_shape();
  std::vector<std::string> constant_values(ng::shape_size(shape), "3");
  auto ng_exponent =
      ConstructNgNode<ng::op::Constant>(op->name(), et, shape, constant_values);

  // Raise each element of the input to the power 3.
  auto ng_pow =
      ConstructNgNode<ng::op::Power>(op->name(), ng_input, ng_exponent);

  // Create a constant tensor populated with the value -1/2.
  std::vector<std::string> constant_diff(ng::shape_size(shape), "-0.5");
  auto ng_diff =
      ConstructNgNode<ng::op::Constant>(op->name(), et, shape, constant_diff);
  auto ng_result = ConstructNgNode<ng::op::Multiply>(
      op->name(),
      (ConstructNgNode<ng::op::Multiply>(op->name(), ng_pow, ng_delta)),
      ng_diff);
  SaveNgOp(ng_op_map, op->name(), ng_result);
  return Status::OK();
}

static Status TranslateShapeOp(const Node* op,
                               const std::vector<const Tensor*>&,
                               Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  // the shape of the input tensor which will be the value to the Constant Op
  auto input_shape = ng_input->get_shape();

  // the rank of the input tensor which will be the shape to the Constant Op
  size_t rank = input_shape.size();

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "out_type", &dtype));

  // the inputs to the Constant Op
  ng::element::Type type;
  TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(dtype, &type));

  auto shape = ng::Shape(1, rank);

  std::vector<int> values(rank);
  for (size_t i = 0; i < rank; i++) {
    values[i] = input_shape[i];
  }
  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<ng::op::Constant>(op->name(), type, shape, values));
  return Status::OK();
}

static Status TranslateSigmoidGradOp(const Node* op,
                                     const std::vector<const Tensor*>&,
