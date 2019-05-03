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
#pragma once

#include <string.h>
#include <vector>

#include "ngraph_backend_manager.h"
#include "tensorflow/core/lib/core/errors.h"

using namespace std;

namespace tensorflow {
namespace ngraph_bridge {
namespace config {
extern "C" {
extern void NgraphEnable();
extern void NgraphDisable();
extern bool NgraphIsEnabled();

extern size_t NgraphBackendsLen();
extern bool NgraphListBackends(char** backends, int backends_len);
extern bool NgraphSetBackend(const char* backend);
extern bool NgraphIsSupportedBackend(const char* backend);
extern bool NgraphGetCurrentlySetBackendName(char** backend);

extern void NgraphStartLoggingPlacement();
extern void NgraphStopLoggingPlacement();
extern bool NgraphIsLoggingPlacement();

extern void NgraphSetDisabledOps(const char* op_type_list);
extern const char* NgraphGetDisabledOps();
}
// // TODO: why is this not const?
extern vector<string> ListBackends();
extern std::set<string> GetDisabledOps();

}  // namespace config
}  // namespace ngraph_bridge
}  // namespace tensorflow
