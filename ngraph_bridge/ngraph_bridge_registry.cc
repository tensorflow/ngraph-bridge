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

#include "ngraph_bridge/ngraph_bridge_registry.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"

#ifdef NGRAPH_BRIDGE_STATIC_LIB_ENABLE
namespace tensorflow {

namespace ngraph_bridge {

void register_ngraph_bridge() {
  register_ngraph_ops();
#if defined(NGRAPH_TF_ENABLE_VARIABLES_AND_OPTIMIZERS)
  register_ngraph_enable_variable_ops();
#endif
}
}
}
#endif
