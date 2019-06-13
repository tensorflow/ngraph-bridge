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

#ifndef NGRAPH_TF_BRIDGE_BACKEND_CONFIG_H_
#define NGRAPH_TF_BRIDGE_BACKEND_CONFIG_H_

#include <ostream>

#include "ngraph_log.h"
#include "tensorflow/core/lib/core/errors.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

class BackendConfig {
 public:
  friend class BackendManager;

 private:
  BackendConfig() = delete;

 protected:
  // BackendNNPI Config calls this constructor
  BackendConfig(string backend_name);
  unordered_map<string, string> split(string backend_config);
  vector<string> get_optional_attributes();

  virtual string join(unordered_map<string, string> optional_parameters);
  virtual ~BackendConfig();

  string backend_name_;
  vector<string> optional_attributes_;
};

class BackendNNPIConfig : public BackendConfig {
 public:
  friend class BackendManager;

 private:
  BackendNNPIConfig();
  string join(unordered_map<string, string> optional_parameters);
  virtual ~BackendNNPIConfig();
};

}  // namespace ngraph_bridge
}  // namespace tensorflow

#endif
// NGRAPH_TF_BRIDGE_BACKEND_CONFIG_H