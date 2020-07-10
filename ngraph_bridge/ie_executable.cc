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
    m_infer_req.SetBlob(it.first, tv->get_blob());
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
    m_infer_req.SetBlob(it.first, tv->get_blob());
    i++;
  }

  m_infer_req.Infer();
  return true;
}
}
}