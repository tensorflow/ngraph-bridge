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
  std::string join(std::map<std::string, std::string>);
  std::map<std::string, std::string> split(std::string);

  virtual vector<string> get_optional_attributes();
  ~BackendConfig();

 private:
  vector<string> optional_attributes_ = {"ngraph_device_config"};
};

class BackendNNPIConfig : public BackendConfig {
 public:
  std::string join(std::map<std::string, std::string>);
  vector<string> get_optional_attributes();

  ~BackendNNPIConfig();

 private:
  vector<string> optional_attributes_ = {"ngraph_device_id", "ngraph_ice_cores",
                                         "ngraph_max_batch_size"};
};

}  // namespace ngraph_bridge
}  // namespace tensorflow

#endif
// NGRAPH_TF_BRIDGE_BACKEND_CONFIG_H