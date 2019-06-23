/*******************************************************************************
 * Copyright 2019 Intel Corporation
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

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include <thread>
#include "ngraph/event_tracing.hpp"
#include "ngraph_backend_manager.h"
#include "version.h"

#include "config_setting.h"
#include "inference_engine.h"

using namespace std;
using namespace config_setting;
using namespace infer_multiple_networks;

namespace tf = tensorflow;

extern tf::Status PrintTopLabels(const std::vector<tf::Tensor>& outputs,
                                 const string& labels_file_name);

// Prints the available backends
void PrintAvailableBackends() {
  // Get the list of backends
  auto supported_backends =
      tf::ngraph_bridge::BackendManager::GetSupportedBackendNames();
  vector<string> backends(supported_backends.begin(), supported_backends.end());

  cout << "Available backends: " << endl;
  for (auto& backend_name : backends) {
    cout << "Backend: " << backend_name << std::endl;
  }
}

// Sets the specified backend. This backend must be set BEFORE running
// the computation
tf::Status SetNGraphBackend(const string& backend_name) {
  // Select a backend
  tf::Status status =
      tf::ngraph_bridge::BackendManager::SetBackendName(backend_name);
  return status;
}

void PrintVersion() {
  // nGraph Bridge version info
  std::cout << "Bridge version: " << tf::ngraph_bridge::ngraph_tf_version()
            << std::endl;
  std::cout << "nGraph version: " << tf::ngraph_bridge::ngraph_lib_version()
            << std::endl;
  std::cout << "CXX11_ABI Used: "
            << tf::ngraph_bridge::ngraph_tf_cxx11_abi_flag() << std::endl;
  std::cout << "Grappler Enabled? "
            << (tf::ngraph_bridge::ngraph_tf_is_grappler_enabled()
                    ? std::string("Yes")
                    : std::string("No"))
            << std::endl;
  std::cout << "Variables Enabled? "
            << (tf::ngraph_bridge::ngraph_tf_are_variables_enabled()
                    ? std::string("Yes")
                    : std::string("No"))
            << std::endl;

  PrintAvailableBackends();
}

int main(int argc, char** argv) {
  string json_file = "data/template1.json";

  std::vector<tf::Flag> flag_list = {
      tf::Flag("json_file", &json_file, "config file to be loaded"),
  };

  string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    std::cout << usage;
    return -1;
  }

  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    std::cout << "Error: Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  std::cout << "Component versions\n";
  PrintVersion();

  infer_multiple_networks::InferenceManager* infer_mgr =
      InferenceManager::getInstance();
  infer_mgr->LoadConfig(json_file);
  infer_mgr->LoadNetworks();

  // Start inference
  infer_mgr->Start();

  infer_mgr->WaitForDone();

  std::cout << "Done" << std::endl;
  return 0;
}
