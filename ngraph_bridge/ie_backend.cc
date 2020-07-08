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

#include "ie_backend.h"

#include <ie_core.hpp>
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset.hpp"

using namespace std;
using namespace ngraph;

namespace tensorflow {
namespace ngraph_bridge {

IE_Backend::IE_Backend(const string& configuration_string) {
  string config = configuration_string;
  // Get device name, after colon if present: IE:CPU -> CPU
  auto separator = config.find(":");
  if (separator != config.npos) {
    config = config.substr(separator + 1);
  }
  m_device = config;
}

shared_ptr<Executable> IE_Backend::compile(shared_ptr<Function> func, bool) {
  return make_shared<IE_Executable>(func, m_device);
}

bool IE_Backend::is_supported(const Node& node) const {
  const auto& opset = ngraph::get_opset3();
  return opset.contains_op_type(&node);
}

bool IE_Backend::is_supported_property(const Property) const { return false; }

shared_ptr<runtime::Tensor> IE_Backend::create_dynamic_tensor(
    const element::Type& type, const PartialShape& shape) {
  return make_shared<IETensor>(type, shape);
}

vector<string> IE_Backend::get_registered_devices() {
  InferenceEngine::Core core;
  return core.GetAvailableDevices();
}

shared_ptr<runtime::Tensor> IE_Backend::create_tensor() {
  throw runtime_error("IE_Backend::create_tensor() not supported");
}

shared_ptr<runtime::Tensor> IE_Backend::create_tensor(
    const element::Type& element_type, const Shape& shape) {
  return make_shared<IETensor>(element_type, shape);
}

shared_ptr<runtime::Tensor> IE_Backend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* data) {
  shared_ptr<runtime::Tensor> tensor =
      make_shared<IETensor>(element_type, shape);
  tensor->write(data, shape_size(shape) * element_type.size());
  return tensor;
}
}
}