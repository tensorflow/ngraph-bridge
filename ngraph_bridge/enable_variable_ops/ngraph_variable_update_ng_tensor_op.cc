/*******************************************************************************
 * Copyright 2019 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use thi0s file except in compliance with the License.
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
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/default/logging.h"

#include "ngraph/event_tracing.hpp"
#include "ngraph/runtime/backend.hpp"

#include "ngraph_bridge/enable_variable_ops/ngraph_catalog.h"
#include "ngraph_bridge/enable_variable_ops/ngraph_var.h"
#include "ngraph_bridge/ngraph_freshness_tracker.h"
#include "ngraph_bridge/ngraph_timer.h"
#include "ngraph_bridge/ngraph_utils.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

/* -------------------------------------------------
//
// NGraphVariableUpdateNGTensor
//
---------------------------------------------------*/

class NGraphVariableUpdateNGTensorOp : public OpKernel {
 private:
  int ng_graph_id_;
  string ng_variable_shared_name_;

 public:
  ~NGraphVariableUpdateNGTensorOp() {
    NGRAPH_VLOG(4) << "~NGraphVariableUpdateNGTensorOp::" << name() << endl;
  }

  explicit NGraphVariableUpdateNGTensorOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("ngraph_graph_id", &ng_graph_id_));
    OP_REQUIRES_OK(context, context->GetAttr("ngraph_variable_shared_name", &ng_variable_shared_name_));

    NGRAPH_VLOG(4) << "NGraphVariableUpdateNGTensorOp:: Constructor called for: " << def().name()
                   << " ,Graph ID "<< ng_graph_id_;

    OP_REQUIRES(context, IsRefType(context->input_type(0)),
                errors::InvalidArgument("lhs input needs to be a ref type"));
  }

  void Compute(OpKernelContext* context) override {
    std::ostringstream oss;
    // Start event tracing
    ngraph::Event event_compute(oss.str(), name(), "");

    NGRAPH_VLOG(4) << "NGraphVariableUpdateNGTensorOp:: Compute called for: " << def().name()
                   << " ,Graph ID "<< ng_graph_id_;

    bool ref_exists = NGraphCatalog::ExistsInInputVariableSharedNameMap(
        ng_graph_id_, def().name(), 0);
    if (!ref_exists) {
      OP_REQUIRES(context, ref_exists,
                  errors::Internal(
                      "Caught exception : RefInput to NGraphVariableUpdateNGTensor not found \n"));
    }
    // string get_ref_var_name = NGraphCatalog::GetInputVariableSharedName(
    //     ng_graph_id_, def().name(), 0);

    // Since we have ngraph_variable_shared_name as an attribute, 
    // we can use that to get the variable from the context
    NGraphVar* var;
    OP_REQUIRES_OK(context,
                   context->resource_manager()->Lookup<NGraphVar>(
                       context->resource_manager()->default_container(),
                       ng_variable_shared_name_, &var));

    // We always return the input ref.
    // As per TF
    // Set the output Ref Tensor at output_index to be an alias of the
    // input Ref Tensor at input_index.
    // REQUIRES: IsRefType(input_dtype(input_index)).
    // REQUIRES: IsRefType(output_dtype(output_index)).
    context->forward_ref_input_to_ref_output(0, 0); // do we need this ? I think we do but unsure
    // need to fully understand the purpose

    NGRAPH_VLOG(4) << "NGraphVariableUpdateNGTensorOp:: Updating ng tensor";
    if (var->copy_tf_to_ng()) {
        NGRAPH_VLOG(4) << "NGraphVariableUpdateNGTensorOp:: Updated ng tensor";
        // Is there need to keep track of the number of copies ?
    }

    // Unref Var
    var->Unref();

    // Stop event tracing
    event_compute.Stop();
    ngraph::Event::write_trace(event_compute);
  }
};

REGISTER_KERNEL_BUILDER(Name("NGraphVariableUpdateNGTensor").Device(DEVICE_CPU),
                        NGraphVariableUpdateNGTensorOp);

}  // namespace ngraph_bridge

}  // namespace tensorflow
