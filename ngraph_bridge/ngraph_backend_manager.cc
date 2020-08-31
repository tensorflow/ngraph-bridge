/*******************************************************************************
 * Copyright 2017-2020 Intel Corporation
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

#include "ngraph_bridge/ngraph_backend_manager.h"

#if !defined(ENABLE_OPENVINO)
#include "ngraph/runtime/backend_manager.hpp"
#endif

using namespace std;

namespace tensorflow {
namespace ngraph_bridge {

shared_ptr<Backend> BackendManager::m_backend;
string BackendManager::m_backend_name;
mutex BackendManager::m_backend_mutex;

BackendManager::~BackendManager() {
  NGRAPH_VLOG(2) << "BackendManager::~BackendManager()";
}

Status BackendManager::SetBackend(const string& backend_name) {
  NGRAPH_VLOG(2) << "BackendManager::SetBackend(" << backend_name << ")";
  shared_ptr<Backend> backend;
  auto status = CreateBackend(backend, backend_name);
  if (!status.ok() || backend == nullptr) {
    return errors::Internal("Failed to set backend: ", status.error_message());
  }

  lock_guard<mutex> lock(m_backend_mutex);
  m_backend = backend;
  m_backend_name = backend_name;
  return Status::OK();
}

shared_ptr<Backend> BackendManager::GetBackend(const string& backend_name) {
  NGRAPH_VLOG(2) << "BackendManager::GetBackend(" << backend_name << ")";
  if (m_backend == nullptr) {
    auto status = SetBackend(backend_name);
    if (!status.ok()) {
      NGRAPH_VLOG(1) << "Failed to get backend: " << status.error_message();
      throw errors::Internal("Failed to get backend: ", status.error_message());
    }
  }
  lock_guard<mutex> lock(m_backend_mutex);
  return m_backend;
}

string BackendManager::GetBackendName() {
  NGRAPH_VLOG(2) << "BackendManager::GetBackendName()";
  if (m_backend == nullptr) {
    GetBackend();
  }
  lock_guard<mutex> lock(m_backend_mutex);
  return m_backend_name;
}

Status BackendManager::CreateBackend(shared_ptr<Backend>& backend,
                                     string& backend_name) {
// Register backends for static linking
#if defined(NGRAPH_BRIDGE_STATIC_LIB_ENABLE)
  ngraph_register_cpu_backend();
  ngraph_register_interpreter_backend();
#endif

  auto bname = backend_name;
  const char* env = std::getenv("NGRAPH_TF_BACKEND");
  if (env != nullptr) {
    bname = string(env);
  }

  try {
    backend = Backend::create(bname);
  } catch (const std::exception& e) {
    return errors::Internal("Could not create backend of type ", backend_name,
                            ". Got exception: ", e.what());
  }
  if (backend == nullptr) {
    return errors::Internal("Could not create backend of type ", backend_name,
                            " got nullptr");
  }

  backend_name = bname;
  NGRAPH_VLOG(2) << "BackendManager::CreateBackend(): " << backend_name;
  return Status::OK();
}

void BackendManager::SetConfig(const map<string, string>& config) {
  NGRAPH_VLOG(2) << "BackendManager::SetConfig() " << m_backend_name;
  std::string error;
  if (!m_backend->set_config(config, error)) {
    NGRAPH_VLOG(2) << "BackendManager::SetConfig(): Could not set config. "
                   << error;
  }
}

// Returns the nGraph supported backend names
vector<string> BackendManager::GetSupportedBackends() {
#if !defined(ENABLE_OPENVINO)
  return ngraph::runtime::BackendManager::get_registered_backends();
#else
  return Backend::get_registered_devices();
#endif
}

}  // namespace ngraph_bridge
}  // namespace tensorflow
