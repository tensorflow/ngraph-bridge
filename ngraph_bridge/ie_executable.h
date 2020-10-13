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

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <ie_core.hpp>
#include "ngraph/ngraph.hpp"

#include "ngraph_bridge/ngraph_executable.h"
#include "ngraph_bridge/openvino/ie_executor.h"

using namespace std;

namespace tensorflow {
namespace ngraph_bridge {

// A Inference Engine executable object produced by compiling an nGraph
// function.
class IE_Executable final : public Executable {
 public:
  IE_Executable(shared_ptr<ngraph::Function> func, string device);
  virtual ~IE_Executable() {}
  bool call(const vector<shared_ptr<ngraph::runtime::Tensor>>& outputs,
            const vector<shared_ptr<ngraph::runtime::Tensor>>& inputs) final;

  size_t get_batch_size(size_t input_batch_size, std::string device) const;

 private:
  bool call_trivial(const vector<shared_ptr<ngraph::runtime::Tensor>>& outputs,
                    const vector<shared_ptr<ngraph::runtime::Tensor>>& inputs);
  string m_device;
  // This holds the parameters we insert for functions with no input parameters
  vector<pair<string, shared_ptr<ngraph::runtime::Tensor>>> m_hoisted_params;
  // This keeps track of whether the original function was trivial: either a
  // constant function, an identity function or a zero function
  shared_ptr<ngraph::Function> m_trivial_fn;
  shared_ptr<IE_Executor> m_ie_executor;
  shared_ptr<ngraph::Function> m_ng_func;
};
}
}
