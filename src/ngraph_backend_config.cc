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

#include "ngraph_backend_config.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

string BackendConfig::join(unordered_map<string, string> parameters) {
  NGRAPH_VLOG(0) << "JOIN";
}

unordered_map<string, string> BackendConfig::split(string backend_config) {
  NGRAPH_VLOG(0) << "SPLIT";

  int delimiter_index = backend_config.find(':');
  string backend_name = backend_config.substr(0, delimiter_index);
  NGRAPH_VLOG(3) << "Got Backend Name " << backend_name;

  string device_config = backend_config.substr(delimiter_index);
  NGRAPH_VLOG(3) << "Got Device Config  " << device_config;

  unordered_map<string, string> backend_parameters;
  backend_parameters["ngraph_backend"] = backend_name;
  backend_parameters["ngraph_device_config"] = device_config;

  return backend_parameters;
}

BackendConfig::~BackendConfig() {
  NGRAPH_VLOG(2) << "BackendConfig::~BackendConfig() DONE";
};

BackendNNPIConfig::~BackendNNPIConfig() {
  NGRAPH_VLOG(2) << "BackendNNPIConfig::~BackendNNPIConfig() DONE";
};

vector<string> BackendConfig::get_optional_attributes() {
  return BackendConfig::optional_attributes_;
}

vector<string> BackendNNPIConfig::get_optional_attributes() {
  return BackendNNPIConfig::optional_attributes_;
}

}  // namespace ngraph_bridge
}  // namespace tensorflow