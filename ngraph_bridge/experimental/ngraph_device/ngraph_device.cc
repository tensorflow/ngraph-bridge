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

#include <iostream>
#include <string>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/common_runtime/process_state.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/public/session_options.h"

#include "logging/ngraph_log.h"
namespace tensorflow {

class NGraphDeviceContext : public DeviceContext {
 public:
  // Does not take ownership of streams.
  ~NGraphDeviceContext() override {}

  void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                             Tensor* device_tensor, StatusCallback done,
                             bool sync_dst_compute) const override {
  if (cpu_tensor->NumElements() > 0) {
	        NGRAPH_VLOG(3) << "CopyCPUTensorToDevice "
			                     << reinterpret_cast<const void*>(
							                                 cpu_tensor->tensor_data().data())
					                          << " " << reinterpret_cast<const void*>(
										                                     device_tensor->tensor_data().data())
								                       << " " << cpu_tensor->NumElements();

		      void* src_ptr = const_cast<void*>(DMAHelper::base(cpu_tensor));
		            const int64 total_bytes = cpu_tensor->TotalBytes();
			          void* dst_ptr = DMAHelper::base(device_tensor);
				        memcpy(dst_ptr, src_ptr, total_bytes);

					      NGRAPH_VLOG(3) << "CPU Tensor: " << cpu_tensor->DebugString();
	  done(Status::OK());
	  return;
  }
  NGRAPH_VLOG(3) << "CopyCPUTensorToDevice empty tensor";
  NGRAPH_VLOG(3) << cpu_tensor->DebugString();
  done(Status::OK());
  }

  void CopyDeviceTensorToCPU(const Tensor* device_tensor, StringPiece edge_name,
                             Device* device, Tensor* cpu_tensor,
                             StatusCallback done) override {
  if (device_tensor->NumElements() > 0) {
	        NGRAPH_VLOG(3) << "CopyDeviceTensorToCPU "
			                     << reinterpret_cast<const void*>(
							                                 device_tensor->tensor_data().data())
					                          << " " << reinterpret_cast<const void*>(
										                                     cpu_tensor->tensor_data().data())
								                       << device_tensor->NumElements();
		      NGRAPH_VLOG(3) << device_tensor->DebugString();
		      void* src_ptr = const_cast<void*>(DMAHelper::base(device_tensor));
		            const int64 total_bytes = device_tensor->TotalBytes();
			          void* dst_ptr = DMAHelper::base(cpu_tensor);
				        memcpy(dst_ptr, src_ptr, total_bytes);

					      done(Status::OK());
					            return;
						        }
                       NGRAPH_VLOG(3) << "CopyDeviceTensorToCPU empty tensor";
		           NGRAPH_VLOG(3) << device_tensor->DebugString();
			       done(Status::OK());
  }

  void CopyTensorInSameDevice(const Tensor* input_tensor, Device* device,
                              Tensor* output_tensor,
                              StatusCallback done) const override {
    NGRAPH_VLOG(4) << "CopyTensorInSameDevice: DEVICE: " << device->name();
    *output_tensor = *input_tensor;
    done(Status::OK());
  }

  // Not used.
  // void MaintainLifetimeOnStream(const Tensor* t,
  //                              se::Stream* stream) const override {}

  // Status ThenExecute(Device* device, se::Stream* stream,
  //                   std::function<void()> func) override;
};

class NGraphDevice : public LocalDevice {
 public:
  NGraphDevice(const SessionOptions& options,
               const DeviceAttributes& attributes)
      : LocalDevice(options, attributes) {
    NGRAPH_VLOG(4) << "NGraphDevice::ctor CALLED";
  }

  Status Sync() override { return Status::OK(); }

  Allocator* GetAllocator(AllocatorAttributes attr) override {
    // QCHECK(false) << "xpu device allocator not implemented.";
    return ProcessState::singleton()->GetCPUAllocator(0);
  }

  Status FillContextMap(const Graph* graph,
                        DeviceContextMap* device_context_map) override {
    static NGraphDeviceContext* ctx = new NGraphDeviceContext;
    device_context_map->resize(graph->num_node_ids());
    for (Node* n : graph->nodes()) {
      ctx->Ref();
      (*device_context_map)[n->id()] = ctx;
    }
    return Status::OK();
  }
  Status MakeTensorFromProto(const TensorProto& tensor_proto,
                             const AllocatorAttributes alloc_attrs,
                             Tensor* tensor) override {
    Tensor parsed(tensor_proto.dtype());
    if (!parsed.FromProto(cpu_allocator(), tensor_proto)) {
      return errors::InvalidArgument("Cannot parse tensor from tensor_proto.");
    }
    *tensor = parsed;
    return Status::OK();
  }
};

class NGraphDeviceFactory : public DeviceFactory {
 private:
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override {
    devices->emplace_back(new NGraphDevice(
        options,
        Device::BuildDeviceAttributes(name_prefix + "/device:NGRAPH:0",
                                      "NGRAPH", static_cast<Bytes>(2 << 30),
                                      DeviceLocality{}, "NGRAPH Device")));
    return Status::OK();
  }
  // For a specific device factory list all possible physical devices.
  Status ListPhysicalDevices(std::vector<string>* devices) override {
    devices->push_back("/physical_device:NGRAPH:0");
    return Status::OK();
  }
};

REGISTER_LOCAL_DEVICE_FACTORY("NGRAPH", NGraphDeviceFactory, 210);

}  // namespace tensorflow
