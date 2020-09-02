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

#include "logging/ngraph_log.h"
#include "ngraph_bridge/default_opset.h"
#include "ngraph_bridge/ie_executable.h"
#include "ngraph_bridge/ie_tensor.h"
#include "ngraph_bridge/ngraph_utils.h"

using namespace std;
using namespace ngraph;

namespace tensorflow {
namespace ngraph_bridge {

IE_Executable::IE_Executable(shared_ptr<Function> func, string device)
    : m_device{device} {
  NGRAPH_VLOG(2) << "Checking for unsupported ops in IE backend";
  const auto& opset = ngraph::get_opset3();
  for (const auto& node : func->get_ops()) {
    if (!opset.contains_op_type(node.get())) {
      NGRAPH_VLOG(0) << "UNSUPPORTED OP DETECTED: "
                     << node->get_type_info().name;
      THROW_IE_EXCEPTION << "Detected op not belonging to opset3!";
    }
  }

  set_parameters_and_results(*func);
  m_results_orig = m_results;
  m_params_orig = m_parameters;
  InfoSaveResultMaps();
  shared_ptr<Function> func2 = StripOffUnhandledNodes(func);

  HandleNoParamsCase(func2);

  NGRAPH_VLOG(2) << "Creating IE CNN network using nGraph function";
  m_network = InferenceEngine::CNNNetwork(func2);

  if (std::getenv("NGRAPH_TF_DUMP_GRAPHS")) {
    auto& name = m_network.getName();
    m_network.serialize(name + ".xml", name + ".bin");
  }

  if (std::getenv("NGRAPH_TF_DUMP_GRAPHS")) {
    auto func = m_network.getFunction();
    string fn_name = func->get_friendly_name();
    NgraphSerialize("cnn_func1_" + fn_name + ".json", func);
  }

  InfoSaveInOutIndexMaps();

  NGRAPH_VLOG(2) << "Loading IE CNN network to device " << m_device 
                 << ", ng-function="
                 << m_network.getFunction()->get_friendly_name();

  InferenceEngine::Core ie;
  // Load network to the plugin (m_device) and create an infer request
  InferenceEngine::ExecutableNetwork exe_network;
  try {
    exe_network = ie.LoadNetwork(m_network, m_device);
    NGRAPH_VLOG(5) << "IE LoadNetwork done";
  } catch(const std::exception& e) {
      THROW_IE_EXCEPTION << "Exception in IE LoadNetwork: " << e.what();
  }
  m_infer_req = exe_network.CreateInferRequest();
  NGRAPH_VLOG(5) << "IE CreateInferRequest done";

  if (std::getenv("NGRAPH_TF_DUMP_GRAPHS")) {
    auto func = m_network.getFunction();
    string fn_name = func->get_friendly_name();
    NgraphSerialize("cnn_func2_" + fn_name + ".json", func);
  }
  NGRAPH_VLOG(5) << "IE ctor END";
}

bool IE_Executable::call(const vector<shared_ptr<runtime::Tensor>>& outputs,
                         const vector<shared_ptr<runtime::Tensor>>& inputs) {
  NGRAPH_VLOG(5) << "IE call BEGIN";
  // Check if the number of inputs that the CNN network expects is equal to the
  // sum of the
  // inputs specified and the inputs we hoisted, if any.
  InferenceEngine::InputsDataMap input_info = m_network.getInputsInfo();
  if (input_info.size() != (inputs.size() + m_hoisted_params.size())) {
    THROW_IE_EXCEPTION
        << "Function inputs number differ from number of given inputs";
  }
  if (m_params_orig.size() != inputs.size()) {
    THROW_IE_EXCEPTION
        << "Ng-Function inputs number differ from number of given inputs";
  }
  //  Prepare input blobs
  auto func = m_network.getFunction();
  auto parameters = func->get_parameters();
  for (int i = 0; i < inputs.size(); i++) {
    shared_ptr<IETensor> tv = static_pointer_cast<IETensor>(inputs[i]);
    m_infer_req.SetBlob(parameters[i]->get_friendly_name(), tv->get_blob());
  }

  for (const auto& it : m_hoisted_params) {
    shared_ptr<IETensor> tv = static_pointer_cast<IETensor>(it.second);
    m_infer_req.SetBlob(it.first, tv->get_blob());
  }

  InferenceEngine::OutputsDataMap output_info = m_network.getOutputsInfo();
  if (output_info.size() != outputs.size()) {
    THROW_IE_EXCEPTION
        << "Function outputs number differ from number of given outputs";
  }
  NGRAPH_CHECK(m_results_orig.size() == outputs.size(),
               "Mismatching number of output items between tensors (",
               outputs.size(), ") and results (", m_results_orig.size(), ")");

  // Shortcut the Const -> Result values
  for (const auto& it : m_map_cnnconstresult_to_ngnodeptr) {
    auto output_name = it.first;
    const ngraph::op::Constant* const_node = (ngraph::op::Constant*)(it.second);
    if (const_node) {
      int idx_tensor_output = m_map_cnnresult_to_tfidx.at(output_name);
      auto num_bytes = shape_size(const_node->get_shape()) *
                       const_node->get_element_type().size();
#if 0  // DBG_ONLY
      std::cout << "    ... shortcut value from Const node " << output_name
                << " to TF Output " << m_nongraph_const_outputs.at(output_name)
                << ", index="
                << ", num_bytes=" << num_bytes << "\n";
#endif
      const void* value = const_node->get_data_ptr();
      outputs[idx_tensor_output]->write(value, num_bytes);
    } else {
      THROW_IE_EXCEPTION << "\n!!! Cannot get const_node = " << output_name
                         << " for value shortcut !!!\n";
    }
  }

  //  Prepare output blobs
  auto results = func->get_results();
  for (int i = 0; i < outputs.size(); i++) {
    shared_ptr<IETensor> tv = static_pointer_cast<IETensor>(outputs[i]);
    // Since IE has no "result" nodes, we set the blob corresponding to the
    // parent of this result node
    auto parent = results[i]->input_value(0).get_node_shared_ptr();
    m_infer_req.SetBlob(parent->get_friendly_name(), tv->get_blob());
  }

  m_infer_req.Infer();

  NGRAPH_VLOG(5) << "IE call END";
  return true;
}


void IE_Executable::HandleNoParamsCase(shared_ptr<ngraph::Function>& func) {
  NGRAPH_VLOG(2) << "Checking for function parameters in IE backend";
  if (func->get_parameters().size() == 0) {
    NGRAPH_VLOG(1) << "No parameters found in nGraph function!";
    // Try to find a node that can be converted into a "static input"
    bool param_replaced = false;
    for (const auto& node : func->get_ordered_ops()) {
      // Only try to convert constant nodes at the edge to parameters
      // FIXME: IE cannot handle input parameters with i64/u6 precision
      // at the moment
      if (node->get_input_size() == 0 && node->is_constant() &&
          !(node->get_element_type() == ngraph::element::i64 ||
            node->get_element_type() == ngraph::element::u64)) {
        auto constant = ngraph::as_type_ptr<opset::Constant>(node);
        auto element_type = constant->get_element_type();
        auto shape = constant->get_shape();
        auto param = std::make_shared<opset::Parameter>(element_type, shape);
        param->set_friendly_name(node->get_friendly_name());
        ngraph::replace_node(node, param);
        // nGraph doesn't provide a way to set a parameter to an existing
        // function, so we clone the function here...
        auto saved_func_friendly_name = func->get_friendly_name();
        func = make_shared<Function>(func->get_results(), ParameterVector{param});
        func->set_friendly_name(saved_func_friendly_name);
        auto ie_tensor = make_shared<IETensor>(element_type, shape);
        ie_tensor->write(constant->get_data_ptr(),
                         shape_size(shape) * element_type.size());
        m_hoisted_params.push_back(
            make_pair(param->get_friendly_name(), ie_tensor));
        NGRAPH_VLOG(1) << "Converted node " << constant << " to a parameter "
                       << param;
        param_replaced = true;
        break;
      }
      if (!param_replaced) {
        THROW_IE_EXCEPTION
            << "Unable to add a parameter to a function with no parameters!";
      }
    }
  }
}


void IE_Executable::InfoSaveResultMaps() {
  // We need to save the mapping from CNN network's Result nodes to the original
  // NG-function's result-contributing nodes
  // m_map_result_to_ngnode e.g. (result, from) e.g. Result_353->Constant_673,
  // Result_350->ngraph_output_1
  // Also...
  // OPV/IE CNN doesn't handle Constant nodes that are directly connected to an
  // Output/Result node, save in m_nongraph_const_outputs
  // (input-const, output-result) e.g. Constant_675->Result_352,
  // ngraph_output_1->Result_350
  for (const auto& ng_result_node :
       m_results) {  // Note: order of m_results matches with ngfunc's outputs'
                     // order, and TF-output-tensors
    const auto& ng_result_name =
        ng_result_node->get_name();  // Result_353 or Result_350
    // Find within the related NGFUNC, the parent/contributing node which feeds
    // to this cnn_result_name
    const auto& cnn_result_node =
        ng_result_node->input(0).get_source_output().get_node();
    const auto& cnn_result_name =
        cnn_result_node->get_friendly_name();  //  e.g. Constant_675 or
                                               //  ngraph_output_1 // Don't use
                                               //  get_name()!!
#if 0                                          // DBG_ONLY
    std::cout << " IE_Executable cnn --> ng_result : " <<
    cnn_result_name << " --> " << ng_result_name << "\n";
#endif
    m_map_result_to_ngnode.insert(
        std::pair<std::string, std::string>(ng_result_name, cnn_result_name));
    if (cnn_result_node->is_constant()) {
      // (input-const, output-result) e.g. Constant_675->Result_352,
      // ngraph_output_1->Result_350
      m_nongraph_const_outputs.insert(
          std::pair<std::string, std::string>(cnn_result_name, ng_result_name));
      const ngraph::op::Constant* const_node =
          dynamic_cast<const ngraph::op::Constant*>(&(*cnn_result_node));
      if (const_node) {
        m_map_cnnconstresult_to_ngnodeptr.insert(
            std::pair<std::string, void*>(cnn_result_name, (void*)const_node));
      } else {
        THROW_IE_EXCEPTION << "\n!!! Cannot dynamic_cast<const "
                              "ngraph::op::Constant*>, const-node = "
                           << cnn_result_name << " !!!\n";
      }
    }
  }

// Example: Result_353->Constant_673, Result_350->ngraph_output_1, ...
#if 0  // DBG_ONLY
  std::cout << "\n*** m_results " << m_results.size() << " ==> ";
  for (auto& op : m_results) { std::cout << op->get_name() << ", "; }
  std::cout << "\n";
  std::cout << "\n*** m_map_result_to_ngnode " <<
  m_map_result_to_ngnode.size() << " ==> "; for (auto const& pair :
  m_map_result_to_ngnode) { std::cout << pair.first << "->" << pair.second <<
  ", "; } std::cout << "\n";
#endif
#if 0  // DBG_ONLY
  std::cout << "\n*** m_nongraph_const_outputs " <<
  m_nongraph_const_outputs.size() << " ==> "; for (auto const& pair :
  m_nongraph_const_outputs) { std::cout << pair.first << "->" << pair.second
  << ", "; } std::cout << "\n";
#endif
  // Example: Constant_673->Result_353,  ngraph_output_1->Result_350, ...

  NGRAPH_CHECK(m_results.size() == m_map_result_to_ngnode.size(),
               "Mismatching number of result/output items");
  for (auto& op : m_results) {
    if (m_map_result_to_ngnode.count(op->get_name()) == 0) {
      std::cout << "m_results ==> ";
      for (auto& op2 : m_results) {
        std::cout << op2->get_name() << ", ";
      }
      std::cout << "\n";
      THROW_IE_EXCEPTION << "\n!!! m_results op " << op->get_name()
                         << " not found in m_map_result_to_ngnode !!!\n";
    }
  }
}

shared_ptr<Function> IE_Executable::StripOffUnhandledNodes(
    const shared_ptr<Function>& func) {
  shared_ptr<Function> func2 = func;

  // Let's don't send disconnected Const -> Result nodes to IE
  if (m_nongraph_const_outputs.size() > 0) {
    vector<shared_ptr<ngraph::Node>>
        ng_result_list2;  // (func->get_results().size() -
                          // m_nongraph_const_outputs.size());
    for (auto& opresult : func->get_results()) {
      const auto& ng_result_name = opresult->get_name();
      if (m_map_result_to_ngnode.count(ng_result_name) == 1) {
        const auto& cnn_result_name = m_map_result_to_ngnode.at(ng_result_name);
        if (m_nongraph_const_outputs.count(cnn_result_name) == 1) {
          continue;
        }
      }
      ng_result_list2.push_back(opresult);
    }
    func2 =
        make_shared<ngraph::Function>(ng_result_list2, func->get_parameters());
    func2->set_friendly_name(func->get_friendly_name());
  }

  // after above
  // Let's strip off any isolated Param nodes
  ngraph::ParameterVector ng_param_list2;
  for (auto& node : func2->get_parameters()) {  // node is a shared_ptr of Node
    if (node->get_output_size() > 0 &&
        node->output(0).get_target_inputs().size() == 0) {
      NGRAPH_VLOG(5) << "!! Stripping off isolated Param node: "
                     << node->get_friendly_name();
    } else {
      ng_param_list2.push_back(node);
    }
  }
  func2 = make_shared<ngraph::Function>(func2->get_results(), ng_param_list2);
  func2->set_friendly_name(func->get_friendly_name());

  return func2;
}

void IE_Executable::InfoSaveInOutIndexMaps() {
  // Save the input index mappings from CNN's param name to TF/NGraph's input
  // index
  for (const auto& node :
       m_network.getFunction()->get_ops()) {  // node is a shared_ptr of Node
    if (node->is_parameter()) {
      // const ngraph::op::Parameter* param_nodeptr = dynamic_cast<const
      // ngraph::op::Parameter*>(&(*node));
      // const std::shared_ptr<ngraph::op::Parameter> *param_nodeptr =
      // dynamic_cast<const std::shared_ptr<ngraph::op::Parameter> *>(&node));
      // std::shared_ptr<ngraph::op::Parameter>& param_node =
      const auto& param_node = as_type_ptr<ngraph::op::Parameter>(node);
      if (param_node) {
        int idx = (int)m_network.getFunction()->get_parameter_index(param_node);
        m_map_cnnparam_to_tfidx.insert(
            std::pair<std::string, int>(node->get_friendly_name(), idx));
      } else {
        THROW_IE_EXCEPTION << "\n!!! Cannot dynamic_cast parameter node = "
                           << node->get_friendly_name() << " !!!\n";
      }
    }
  }

#if 0  // DBG_ONLY
  std::cout << "\nIE_Executable ctor, m_map_cnnparam_to_tfidx " <<
  m_network.getFunction()->get_friendly_name() << " ==> "; for (auto const&
  pair : m_map_cnnparam_to_tfidx) { std::cout << pair.first << "->" <<
  pair.second << ", "; } std::cout << "\n";
#endif
  ////Dont' check -->
  /// NGRAPH_CHECK(m_map_cnnparam_to_tfidx.size()==func->get_parameters().size(),
  ///"Mismatching number of param/input items for orig-func and
  /// m_network.getFunction()");

  // Save the output index mappings from CNN's result name to TF tensor's output
  // index
  // Example: same numbers of items in each (e.g. 10)
  // m_network.getOutputsInfo() => Add_359(*), Constant_673, Constant_675,
  // ngraph_output_0(*), ngraph_output_1, ngraph_output_2, ngraph_output_6,
  // ngraph_output_7, ngraph_output_8, ngraph_output_9
  // m_results = Result_349 (0), Result_350 (1), Result_351 (2), Result_352 (3),
  // Result_353 (4), Result_354 (5), Result_355 (6), Result_356 (7), Result_357
  // (8), Result_358 (9),
  // Also, order of Output TF tensors follow the order of m_results, but *NOT*
  // the order of m_network.getOutputsInfo()
  NGRAPH_CHECK(m_results.size() >= m_network.getOutputsInfo().size(),
               "Mismatching number of output items");
  NGRAPH_CHECK(m_results_orig.size() >= m_results.size(),
               "m_results_orig must be >= m_results");
  int idx = 0;
  for (auto aNodeShPtr : m_results_orig) {
    string ng_result_name = aNodeShPtr->get_name();  // e.g. Result_350
    if (m_map_result_to_ngnode.find(ng_result_name) ==
        m_map_result_to_ngnode.end()) {
      THROW_IE_EXCEPTION
          << "\n!!! Cannot locate in m_map_result_to_ngnode, ng_result_name = "
          << ng_result_name << " !!!\n";
    }
    string output_name =
        m_map_result_to_ngnode.at(ng_result_name);  // e.g. Constant_673
#if 0                                               // DBG_ONLY
    std::cout << "\nIn IE_Executable::call Prepare output blobs, output_name = " <<
      output_name << ", ng_result_name = " << ng_result_name <<
      ", idx=" << idx << ", tensor outputs[idx].size()=" <<
      outputs[idx]->get_size_in_bytes() << "\n";
#endif

    m_map_cnnresult_to_tfidx.insert(
        std::pair<std::string, int>(output_name, idx));

    idx++;
  }
#if 0  // DBG_ONLY
  std::cout << "\nIE_Executable ctor, m_map_cnnresult_to_tfidx " <<
  m_network.getFunction()->get_friendly_name() << " ==> "; for (auto const&
  pair : m_map_cnnresult_to_tfidx) { std::cout << pair.first << "->" <<
  pair.second << ", "; } std::cout << "\n";
#endif
}


}
}
