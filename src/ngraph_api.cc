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

#include "ngraph_api.h"

namespace ng = ngraph;

namespace tensorflow {
namespace ngraph_bridge {
namespace config {

static bool _is_enabled = true;
static bool _is_logging_placement = false;
static std::set<std::string> disabled_op_types{};

extern "C" {
void NgraphEnable() { _is_enabled = true; }
void NgraphDisable() { _is_enabled = false; }
bool NgraphIsEnabled() { return _is_enabled; }

size_t NgraphBackendsLen() {
  return BackendManager::GetNumOfSupportedBackends();
}

bool NgraphListBackends(char** backends, int backends_len) {
  const auto ngraph_backends = ListBackends();
  if (backends_len != ngraph_backends.size()) {
    return false;
  }

  for (size_t idx = 0; idx < backends_len; idx++) {
    backends[idx] = strdup(ngraph_backends[idx].c_str());
  }
  return true;
}

bool NgraphSetBackend(const char* backend) {
  if (BackendManager::SetBackendName(backend) != tensorflow::Status::OK()) {
    return false;
  }
  return true;
}

extern bool NgraphIsSupportedBackend(const char* backend) {
  return BackendManager::IsSupportedBackend(backend);
}

extern bool NgraphGetCurrentlySetBackendName(char** backend) {
  string backend_set = "";
  backend_set = BackendManager::GetCurrentlySetBackendName();
  backend[0] = strdup(backend_set.c_str());
  return true;
}

void NgraphStartLoggingPlacement() { _is_logging_placement = true; }
void NgraphStopLoggingPlacement() { _is_logging_placement = false; }
bool NgraphIsLoggingPlacement() {
  return _is_enabled && (_is_logging_placement ||
                         std::getenv("NGRAPH_TF_LOG_PLACEMENT") != nullptr);
}

extern void NgraphSetDisabledOps(const char* op_type_list) {
  auto disabled_ops_list = ng::split(std::string(op_type_list), ',');
  // In case string is '', then splitting yields ['']. So taking care that ['']
  // corresponds to empty set {}
  if (disabled_ops_list.size() >= 1 && disabled_ops_list[0] != "") {
    disabled_op_types =
        set<string>(disabled_ops_list.begin(), disabled_ops_list.end());
  } else {
    disabled_op_types = {};
  }
}

extern const char* NgraphGetDisabledOps() {
  return ng::join(GetDisabledOps(), ",").c_str();
}
}

// note that TensorFlow always uses camel case for the C++ API, but not for
// Python

// Keeping these C++ functions because of the data structures.
vector<string> ListBackends() {
  auto supported_backends = BackendManager::GetSupportedBackendNames();
  vector<string> backend_list(supported_backends.begin(),
                              supported_backends.end());
  return backend_list;
}

std::set<string> GetDisabledOps() { return disabled_op_types; }

}  // namespace config
}  // namespace ngraph_bridge
}  // namespace tensorflow
