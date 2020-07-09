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

#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset.hpp"
#include "ngraph/pass/algebraic_simplification.hpp"
#include "ngraph/pass/get_output_element_elimination.hpp"
#include "ngraph/pass/like_replacement.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/nop_elimination.hpp"
#include "ngraph/pass/opset1_upgrade.hpp"
#include "ngraph/pass/reshape_elimination.hpp"
#include "ngraph/pass/reshape_sinking.hpp"
#include "ngraph/pass/zero_dim_tensor_elimination.hpp"

#include "ngraph_bridge/ie_executable.h"
#include "ngraph_bridge/ie_tensor.h"

using namespace std;
using namespace ngraph;

namespace tensorflow {
namespace ngraph_bridge {

namespace {
InferenceEngine::Blob::Ptr fill_blob(InferenceEngine::SizeVector shape,
                                     const void* data, size_t data_size,
                                     const element::Type& elem_type) {
  InferenceEngine::Layout layout;
  switch (shape.size()) {
    case 0:
      layout = InferenceEngine::Layout::SCALAR;
      break;
    case 1:
      layout = InferenceEngine::Layout::C;
      break;
    case 2:
      layout = InferenceEngine::Layout::NC;
      break;
    case 3:
      layout = InferenceEngine::Layout::CHW;
      break;
    case 4:
      layout = InferenceEngine::Layout::NCHW;
      break;
    case 5:
      layout = InferenceEngine::Layout::NCDHW;
      break;
    case 6:
      layout = InferenceEngine::Layout::GOIDHW;
      break;
    default:
      THROW_IE_EXCEPTION << "Can't convert dims " << shape.size()
                         << " to Layout!";
  }

  InferenceEngine::MemoryBlob::Ptr blob;

  auto size = data_size * elem_type.size();

#define MAKE_IE_TBLOB(type_, precision_, shape_, layout_)                 \
  make_shared<InferenceEngine::TBlob<type_>>(                             \
      InferenceEngine::TensorDesc{InferenceEngine::Precision::precision_, \
                                  shape_, layout_},                       \
      (type_*)data, size)

  switch (elem_type) {
    case element::Type_t::f32:
      blob = MAKE_IE_TBLOB(float, FP32, shape, layout);
      break;
    case element::Type_t::i16:
      blob = MAKE_IE_TBLOB(int16_t, I16, shape, layout);
      break;
    case element::Type_t::u8:
      blob = MAKE_IE_TBLOB(uint8_t, U8, shape, layout);
      break;
    case element::Type_t::i8:
      blob = MAKE_IE_TBLOB(int8_t, I8, shape, layout);
      break;
    case element::Type_t::u16:
      blob = MAKE_IE_TBLOB(uint16_t, U16, shape, layout);
      break;
    case element::Type_t::i32:
      blob = MAKE_IE_TBLOB(int32_t, I32, shape, layout);
      break;
    case element::Type_t::i64:
      blob = MAKE_IE_TBLOB(int64_t, I64, shape, layout);
      break;
    case element::Type_t::u64:
      blob = MAKE_IE_TBLOB(uint64_t, U64, shape, layout);
      break;
    case element::Type_t::boolean:
      blob = MAKE_IE_TBLOB(uint8_t, BOOL, shape, layout);
      break;
    default:
      THROW_IE_EXCEPTION << "Can't convert type " << elem_type
                         << " to IE Precision!";
  }
#undef MAKE_IE_TBLOB
  return blob;
}
}

IE_Executable::IE_Executable(shared_ptr<Function> func, string device)
    : m_device{device} {
  const auto& opset = ngraph::get_opset3();
  ngraph::pass::Manager passes;
  passes.register_pass<ngraph::pass::LikeReplacement>();
  passes.register_pass<ngraph::pass::NopElimination>();
  passes.register_pass<ngraph::pass::ZeroDimTensorElimination>();
  passes.register_pass<ngraph::pass::AlgebraicSimplification>();
  passes.register_pass<ngraph::pass::ReshapeSinking>();
  passes.register_pass<ngraph::pass::ReshapeElimination>();
  passes.register_pass<ngraph::pass::RecurrentReshapeElimination>();
  passes.register_pass<ngraph::pass::GetOutputElementElimination>();
  passes.run_passes(func);

  for (const auto& node : func->get_ops()) {
    if (!opset.contains_op_type(node.get())) {
      cout << "UNSUPPORTED OP DETECTED: " << node->get_type_info().name << endl;
      THROW_IE_EXCEPTION << "Detected op not belonging to opset3!";
    }
  }

  m_network = InferenceEngine::CNNNetwork(func);
  set_parameters_and_results(*func);

  InferenceEngine::Core ie;
  // Load model to the plugin (BACKEND_NAME)
  InferenceEngine::ExecutableNetwork exe_network =
      ie.LoadNetwork(m_network, m_device);
  // Create infer request
  m_infer_req = exe_network.CreateInferRequest();
}

bool IE_Executable::call(const vector<shared_ptr<runtime::Tensor>>& outputs,
                         const vector<shared_ptr<runtime::Tensor>>& inputs) {
  InferenceEngine::InputsDataMap input_info = m_network.getInputsInfo();
  if (input_info.size() != inputs.size()) {
    THROW_IE_EXCEPTION
        << "Function inputs number differ from number of given inputs";
  }

  size_t i = 0;
  for (const auto& it : input_info) {
    shared_ptr<IETensor> tv = static_pointer_cast<IETensor>(inputs[i]);
    m_infer_req.SetBlob(
        it.first,
        fill_blob(it.second->getTensorDesc().getDims(), tv->get_data_ptr(),
                  tv->get_element_count(), tv->get_element_type()));
    i++;
  }

  //  Prepare output blobs
  InferenceEngine::OutputsDataMap output_info = m_network.getOutputsInfo();
  if (output_info.size() != outputs.size()) {
    THROW_IE_EXCEPTION
        << "Function outputs number differ from number of given outputs";
  }

  i = 0;
  for (const auto& it : output_info) {
    shared_ptr<IETensor> tv = static_pointer_cast<IETensor>(outputs[i]);
    m_infer_req.SetBlob(
        it.first,
        fill_blob(it.second->getTensorDesc().getDims(), tv->get_data_ptr(),
                  tv->get_element_count(), tv->get_element_type()));
    i++;
  }

  m_infer_req.Infer();
  return true;
}
}
}