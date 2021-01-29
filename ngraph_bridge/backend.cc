//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "backend.h"
#include "default_opset.h"
#include "log.h"

#include <ie_core.hpp>
#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;

namespace tensorflow {
namespace ngraph_bridge {

Backend::Backend(const string& config) {
  string device = config.substr(0, config.find(":"));
  InferenceEngine::Core core;
  auto devices = core.GetAvailableDevices();
  // TODO: Handle multiple devices
  if (find(devices.begin(), devices.end(), device) == devices.end()) {
    stringstream ss;
    ss << "Device '" << config << "' not found.";
    throw runtime_error(ss.str());
  }
  m_device = config;
}

shared_ptr<Executable> Backend::Compile(shared_ptr<ngraph::Function> func,
                                        bool) {
  return make_shared<Executable>(func, m_device);
}

static std::map<std::string, std::set<shared_ptr<ngraph::Node>>>
    TFtoNgraphOpMap{
        {"Abs", {std::make_shared<opset::Abs>()}},
        {"Acos", {std::make_shared<opset::Acos>()}},
        {"Acosh", {std::make_shared<opset::Acosh>()}},
        {"Add", {std::make_shared<opset::Add>()}},
        {"AddN", {std::make_shared<opset::Add>()}},
        {"AddV2", {std::make_shared<opset::Add>()}},
        {"Any", {std::make_shared<opset::ReduceLogicalOr>()}},
        {"All", {std::make_shared<opset::ReduceLogicalAnd>()}},
        {"ArgMax",
         {std::make_shared<opset::TopK>(), std::make_shared<opset::Squeeze>()}},
        {"ArgMin",
         {std::make_shared<opset::TopK>(), std::make_shared<opset::Squeeze>()}},
        {"Asin", {std::make_shared<opset::Asin>()}},
        {"Asinh", {std::make_shared<opset::Asinh>()}},
        {"Atan", {std::make_shared<opset::Atan>()}},
        {"Atanh", {std::make_shared<opset::Atanh>()}},
        {"AvgPool", {std::make_shared<opset::AvgPool>()}},
        {"BiasAdd",
         {std::make_shared<opset::Add>(), std::make_shared<opset::Reshape>()}},
        {"Cast", {std::make_shared<opset::Convert>()}},
        {"Ceil", {std::make_shared<opset::Ceiling>()}},
        {"ConcatV2", {std::make_shared<opset::Concat>()}},
        {"Const", {}},
        {"Conv2D",
         {std::make_shared<opset::Transpose>(),
          std::make_shared<opset::Convolution>()}},
        {"Conv2DBackpropInput",
         {std::make_shared<opset::ConvolutionBackpropData>(),
          std::make_shared<opset::Transpose>()}},
        {"Conv3D",
         {std::make_shared<opset::Convolution>(),
          std::make_shared<opset::Transpose>()}},
        {"Cos", {std::make_shared<opset::Cos>()}},
        {"Cosh", {std::make_shared<opset::Cosh>()}},
        {"Cumsum", {std::make_shared<opset::CumSum>()}},
        {"DepthToSpace", {std::make_shared<opset::DepthToSpace>()}},
        {"DepthwiseConv2dNative",
         {std::make_shared<opset::GroupConvolution>()}},
        {"Equal", {std::make_shared<opset::Equal>()}},
        {"Exp", {std::make_shared<opset::Exp>()}},
        {"ExpandDims", {std::make_shared<opset::Unsqueeze>()}},
        {"Fill", {std::make_shared<opset::Broadcast>()}},
        {"Floor", {std::make_shared<opset::Floor>()}},
        {"FloorDiv",
         {std::make_shared<opset::Divide>(), std::make_shared<opset::Floor>(),
          std::make_shared<opset::Broadcast>()}},
        {"FloorMod", {std::make_shared<opset::FloorMod>()}},
        {"FusedBatchNorm", {std::make_shared<opset::BatchNormInference>()}},
        {"FusedBatchNormV2",
         {std::make_shared<opset::BatchNormInference>(),
          std::make_shared<opset::Transpose>()}},
        {"FusedBatchNormV3",
         {std::make_shared<opset::BatchNormInference>(),
          std::make_shared<opset::Transpose>()}},
        {"Gather", {std::make_shared<opset::Gather>()}},
        {"GatherV2", {std::make_shared<opset::Gather>()}},
        {"_FusedConv2D",
         {std::make_shared<opset::Convolution>(),
          std::make_shared<opset::Minimum>(), std::make_shared<opset::Relu>(),
          std::make_shared<opset::Add>(),
          std::make_shared<opset::BatchNormInference>()}},
        {"_FusedMatMul",
         {std::make_shared<opset::MatMul>(), std::make_shared<opset::Relu>(),
          std::make_shared<opset::Add>(), std::make_shared<opset::Minimum>()}},
        {"Greater", {std::make_shared<opset::Greater>()}},
        {"GreaterEqual", {std::make_shared<opset::GreaterEqual>()}},
        {"Identity", {}},
        {"IsFinite",
         {std::make_shared<opset::NotEqual>(), std::make_shared<opset::Equal>(),
          std::make_shared<opset::LogicalAnd>()}},
        {"L2Loss",
         {std::make_shared<opset::Multiply>(),
          std::make_shared<opset::ReduceSum>(),
          std::make_shared<opset::Divide>()}},
        {"LogSoftmax",
         {std::make_shared<opset::Exp>(), std::make_shared<opset::ReduceMax>(),
          std::make_shared<opset::ReduceSum>(),
          std::make_shared<opset::Subtract>(), std::make_shared<opset::Log>()}},
        {"Less", {std::make_shared<opset::Less>()}},
        {"LessEqual", {std::make_shared<opset::LessEqual>()}},
        {"Log", {std::make_shared<opset::Log>()}},
        {"Log1p",
         {std::make_shared<opset::Add>(), std::make_shared<opset::Log>()}},
        {"LogicalAnd", {std::make_shared<opset::LogicalAnd>()}},
        {"LogicalNot", {std::make_shared<opset::LogicalNot>()}},
        {"LogicalOr", {std::make_shared<opset::LogicalOr>()}},
        {"LRN", {std::make_shared<opset::LRN>()}},
        {"MatMul", {std::make_shared<opset::MatMul>()}},
        {"Max", {std::make_shared<opset::ReduceMax>()}},
        {"Maximum", {std::make_shared<opset::Maximum>()}},
        {"MaxPool",
         {std::make_shared<opset::Transpose>(),
          std::make_shared<opset::MaxPool>()}},
        {"MaxPool3D",
         {std::make_shared<opset::Transpose>(),
          std::make_shared<opset::MaxPool>()}},
        {"Mean", {std::make_shared<opset::ReduceMean>()}},
        {"Min", {std::make_shared<opset::ReduceMin>()}},
        {"Minimum", {std::make_shared<opset::Minimum>()}},
        {"MirrorPad", {std::make_shared<opset::Pad>()}},
        {"Mul", {std::make_shared<opset::Multiply>()}},
        {"Mod", {std::make_shared<opset::Mod>()}},
        {"Neg", {std::make_shared<opset::Negative>()}},
        {"NotEqual", {std::make_shared<opset::NotEqual>()}},
        {"NonMaxSuppressionV2",
         {std::make_shared<opset::NonMaxSuppression>(),
          std::make_shared<opset::Unsqueeze>(),
          std::make_shared<opset::StridedSlice>()}},
        {"OneHot", {std::make_shared<opset::OneHot>()}},
        {"Pack",
         {std::make_shared<opset::Concat>(),
          std::make_shared<opset::Unsqueeze>()}},
        {"Pad", {std::make_shared<opset::Pad>()}},
        {"PadV2", {std::make_shared<opset::Pad>()}},
        {"Pow", {std::make_shared<opset::Power>()}},
        {"Prod", {std::make_shared<opset::ReduceProd>()}},
        {"Range", {std::make_shared<opset::Range>()}},
        {"Rank", {}},
        {"RealDiv", {std::make_shared<opset::Divide>()}},
        {"Reciprocal", {std::make_shared<opset::Power>()}},
        {"Relu", {std::make_shared<opset::Relu>()}},
        {"Relu6", {std::make_shared<opset::Clamp>()}},
        {"Rsqrt", {std::make_shared<opset::Power>()}},
        {"Select", {std::make_shared<opset::Select>()}},
        {"SelectV2", {std::make_shared<opset::Select>()}},
        {"Reshape", {std::make_shared<opset::Reshape>()}},
        {"Shape", {std::make_shared<opset::ShapeOf>()}},
        {"Sigmoid", {std::make_shared<opset::Sigmoid>()}},
        {"Sin", {std::make_shared<opset::Sin>()}},
        {"Sinh", {std::make_shared<opset::Sinh>()}},
        {"Size", {}},
        {"Sign", {std::make_shared<opset::Sign>()}},
        {"Slice", {std::make_shared<opset::StridedSlice>()}},
        {"Snapshot", {}},
        {"Softmax", {std::make_shared<opset::Softmax>()}},
        {"Softplus", {std::make_shared<opset::SoftPlus>()}},
        {"SpaceToDepth", {std::make_shared<opset::SpaceToDepth>()}},
        {"Split", {std::make_shared<opset::Split>()}},
        {"SplitV", {std::make_shared<opset::VariadicSplit>()}},
        {"Sqrt", {std::make_shared<opset::Sqrt>()}},
        {"Square", {std::make_shared<opset::Multiply>()}},
        {"SquaredDifference", {std::make_shared<opset::SquaredDifference>()}},
        {"Squeeze", {std::make_shared<opset::Squeeze>()}},
        {"StridedSlice", {std::make_shared<opset::StridedSlice>()}},
        {"Sub", {std::make_shared<opset::Subtract>()}},
        {"Sum", {std::make_shared<opset::ReduceSum>()}},
        {"Tan", {std::make_shared<opset::Tan>()}},
        {"Tanh", {std::make_shared<opset::Tanh>()}},
        {"Tile", {std::make_shared<opset::Tile>()}},
        {"TopKV2", {std::make_shared<opset::TopK>()}},
        {"Transpose", {std::make_shared<opset::Transpose>()}},
        {"Where",
         {std::make_shared<opset::NonZero>(),
          std::make_shared<opset::Transpose>()}},
        {"Xdivy",
         {std::make_shared<opset::Divide>(), std::make_shared<opset::Equal>(),
          std::make_shared<opset::Select>()}},
        {"Unpack", {std::make_shared<opset::StridedSlice>()}},
        {"ZerosLike", {}},
        {"NoOp", {}},
    };

bool Backend::IsSupported(const char* op) const {
  string op_(op);
  auto ng_op = TFtoNgraphOpMap.find(op_);
  if (ng_op == TFtoNgraphOpMap.end()) {
    NGRAPH_VLOG(0) << "TF Op is not found in the map: " << op;
    return false;
  }

  // Loop through the ngraph op list to query
  const auto& opset = ngraph::get_opset5();
  for (auto it = ng_op->second.begin(); it != ng_op->second.end(); it++) {
    // TODO: check if the given backend/device supports the op. Right now we're
    // assuming
    // that the selected backend supports all opset5 ops
    ngraph::Node& node = **it;
    if (!opset.contains_op_type(&node)) {
      return false;
    }
  }
  return true;
}

}  // namespace ngraph_bridge
}  // namespace tensorflow