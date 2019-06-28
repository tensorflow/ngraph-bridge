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

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

namespace ngraph_bridge {

REGISTER_OP("NGraphVariable")
    .Output("ref: Ref(dtype)")
    .Attr("shape: shape")
    .Attr("dtype: type")
    .Attr("just_looking: bool = false")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ExplicitShape);
    
}  // namespace ngraph_bridge
}  // namespace tensorflow