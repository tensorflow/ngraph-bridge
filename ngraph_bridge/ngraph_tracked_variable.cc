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

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/default/logging.h"

#include "ngraph_bridge/ngraph_utils.h"

namespace tensorflow {

namespace ngraph_bridge {

//
// Forked from tensorflow:tensorflow/core/kernels/variable_ops.{cc,h}
// and tensorflow:tensorflow/core/ops/state_ops.cc.
//

// Resource stored by variables in the resource manager
// (legacy, ref-style version).
//
// (Changes: Renamed from LegacyVar, modified to take a TensorShape in
// constructor.)

// THIS CLASS IS NOT BEING USED ANYWHERE
class NGraphVar : public ResourceBase {
 public:
  explicit NGraphVar(DataType dtype, TensorShape shape)
      : tensor_(dtype, shape) {}
  // Not copyable or movable.
  NGraphVar(const NGraphVar&) = delete;
  NGraphVar& operator=(const NGraphVar&) = delete;

  mutex* mu() { return &mu_; }
  Tensor* tensor() { return &tensor_; }

  string DebugString() const override {
    return strings::StrCat(DataTypeString(tensor_.dtype()), "/",
                           tensor_.shape().DebugString());
  }

 private:
  mutex mu_;
  Tensor tensor_;
  ~NGraphVar() override {}
};

class NGraphVariableOp : public OpKernel {
 public:
  explicit NGraphVariableOp(OpKernelConstruction* context);
  ~NGraphVariableOp() override;
  void Compute(OpKernelContext* ctx) override;

 private:
  TensorShape shape_;
  DataType dtype_;

  mutex init_mu_;
  ContainerInfo cinfo_ GUARDED_BY(init_mu_);
  bool initialized_ GUARDED_BY(init_mu_){false};

  static int s_instance_count;
  int my_instance_id{0};

  TF_DISALLOW_COPY_AND_ASSIGN(NGraphVariableOp);
};

int NGraphVariableOp::s_instance_count = 0;

NGraphVariableOp::NGraphVariableOp(OpKernelConstruction* context)
    : OpKernel(context), dtype_(RemoveRefType(context->output_type(0))) {
  my_instance_id = s_instance_count;
  s_instance_count++;

  OP_REQUIRES_OK(context, context->GetAttr("shape", &shape_));
}

NGraphVariableOp::~NGraphVariableOp() {}

// (Changes: Renamed from VariableOp, modified to pass TensorShape to NGraphVar
// constructor.)
void NGraphVariableOp::Compute(OpKernelContext* ctx) {
  mutex_lock l(init_mu_);
  std::ostringstream oss;
  oss << "NGVariable::Compute::" << name();
  NG_TRACE(oss.str(), name(), "");

  if (!initialized_) {
    OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), def(),
                                    true /* use name() */));
    initialized_ = true;
  }
  auto creator = [this](NGraphVar** var) {
    *var = new NGraphVar(dtype_, shape_);
    //(*var)->tensor()->set_shape(shape_);
    return Status::OK();
  };
  NGraphVar* var;
  OP_REQUIRES_OK(ctx, cinfo_.resource_manager()->LookupOrCreate<NGraphVar>(
                          cinfo_.container(), cinfo_.name(), &var, creator));

  // Output a reference to our tensor, so it may be updated.

  ctx->set_output_ref(0, var->mu(), var->tensor());
  if (ctx->track_allocations() && var->tensor()->IsInitialized()) {
    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    attr.set_nic_compatible(true);
    ctx->record_persistent_memory_allocation(var->tensor()->AllocatedBytes());
  }
  var->Unref();
}

REGISTER_KERNEL_BUILDER(Name("NGraphVariable").Device(DEVICE_CPU),
                        NGraphVariableOp);

}  // namespace ngraph_bridge

}  // namespace tensorflow
