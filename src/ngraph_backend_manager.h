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

// The backend manager class is a singelton class that interfaces with the
// bridge to provide necessary backend

#ifndef NGRAPH_TF_BRIDGE_BACKEND_MANAGER_H_
#define NGRAPH_TF_BRIDGE_BACKEND_MANAGER_H_

#include <atomic>
#include <mutex>
#include <ostream>
#include <vector>

#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph_log.h"
#include "tensorflow/core/lib/core/errors.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

struct Backend {
  shared_ptr<ng::runtime::Backend> backend_ptr;
  mutex backend_mutex;
};

class BackendManager {
 public:
  // Returns the backend name currently set
  // If env variable NGRAPH_TF_BACKEND is set it has precedence
  static Status GetCurrentlySetBackendName(string* backend);

  // Returns the nGraph supported backend names
  static vector<string> GetSupportedBackendNames();

  // Returns number of supported backends
  static size_t GetNumOfSupportedBackends();

  static bool IsSupportedBackend(const string& backend_name);

  static Status SetBackendName(const string& backend_name);

  static Status CreateBackend(const string& backend_name);

  static void ReleaseBackend(const string& backend_name);

  // Returns a backend pointer of the type specified by the backend name
  // The backend must have already been created (use CreateBackend(...))
  static ng::runtime::Backend* GetBackend(const string& backend_name);

  // LockBackend
  static void LockBackend(const string& backend_name);

  // UnlockBackend
  static void UnlockBackend(const string& backend_name);

  ~BackendManager();

 private:
  static string ng_backend_name_;  // currently set backend name
  static mutex ng_backend_name_mutex_;

  // map of cached backend objects
  static map<string, Backend*> ng_backend_map_;
  static mutex ng_backend_map_mutex_;

  // Map of backends and their reference counts
  static std::map<std::string, int> ref_count_each_backend_;
};

}  // namespace ngraph_bridge
}  // namespace tensorflow

#endif
// NGRAPH_TF_BRIDGE_BACKEND_MANAGER_H