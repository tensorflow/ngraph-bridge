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
#include <cstdlib>
#include <mutex>
#include <utility>

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"

#include "ngraph_backend_manager.h"
#include "ngraph_builder.h"
#include "ngraph_cluster_manager.h"
#include "ngraph_encapsulate_impl.h"
#include "ngraph_encapsulate_op.h"
#include "ngraph_log.h"
#include "ngraph_mark_for_clustering.h"
#include "ngraph_timer.h"
#include "ngraph_utils.h"

#include "ngraph/event_tracing.hpp"
#include "ngraph/runtime/backend.hpp"

#if defined NGRAPH_DISTRIBUTED
#include "ngraph/distributed.hpp"
#endif

#if defined(NGRAPH_TF_ENABLE_VARIABLES_AND_OPTIMIZERS)
#include "enable_variable_ops/ngraph_catalog.h"
#include "enable_variable_ops/ngraph_var.h"
#endif

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

//---------------------------------------------------------------------------
//  NGraphEncapsulateImpl::ctor
//---------------------------------------------------------------------------
NGraphEncapsulateImpl::NGraphEncapsulateImpl(string name)
    : m_graph(OpRegistry::Global()),
      m_freshness_tracker(nullptr),
      m_name(name) {}

Status NGraphEncapsulateImpl::ComputeSignature(
    std::vector<Tensor>& input_tensors, std::vector<TensorShape>& input_shapes,
    std::vector<const Tensor*>& static_input_map,
    std::stringstream& signature_ss) {
  // Get the inputs
  for (int i = 0; i < input_tensors.size(); i++) {
    const Tensor& input_tensor = input_tensors[i];
    input_shapes.push_back(input_tensor.shape());
    for (const auto& x : input_tensor.shape()) {
      signature_ss << x.size << ",";
    }
    signature_ss << ";";
  }

  signature_ss << "/";

  static_input_map.resize(input_tensors.size());
  for (int i = 0; i < input_tensors.size(); i++) {
    const Tensor& input_tensor = input_tensors[i];
    if (m_input_is_static[i]) {
      static_input_map[i] = &input_tensor;
      TF_RETURN_IF_ERROR(TensorToStream(signature_ss, input_tensor));
      signature_ss << ";";
    }
  }
  return Status::OK();
}

Status NGraphEncapsulateImpl::GetNgExecutable(
    std::vector<Tensor>& input_tensors,
    const std::pair<string, int64> ctx_params,
    std::vector<TensorShape>& input_shapes,
    std::vector<const Tensor*>& static_input_map,
    ng::runtime::Backend*& op_backend,
    std::shared_ptr<ngraph::runtime::Executable>& ng_exec) {
  std::stringstream signature_ss;
  string signature;

  std::shared_ptr<ngraph::Function> ng_function;
  std::shared_ptr<ngraph::runtime::Executable> evicted_ng_exec;

  NGRAPH_VLOG(4) << "GetNgExec: Got backend of type: " << m_op_backend_name;
  op_backend = BackendManager::GetBackend(m_op_backend_name);

  // Compute Signature
  TF_RETURN_IF_ERROR(ComputeSignature(input_tensors, input_shapes,
                                      static_input_map, signature_ss));
  signature = signature_ss.str();

  if (NGRAPH_VLOG_IS_ON(5)) {
    NGRAPH_VLOG(5) << "Computed signature: " << signature;
  }

  auto it = m_ng_exec_map.find(signature);

  NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute got inputs for cluster "
                 << m_ngraph_cluster;

  // Translate the TensorFlow graph to nGraph.
  if (it == m_ng_exec_map.end()) {
    // Measure the current total memory usage
    long vm, rss, vm0, rss0;
    MemoryProfile(vm0, rss0);

    NGRAPH_VLOG(1) << "Compilation cache miss: " << ctx_params.first;
    TF_RETURN_IF_ERROR(Builder::TranslateGraph(input_shapes, static_input_map,
                                               &m_graph, ng_function));
    ng_function->set_friendly_name(m_name);

    auto function_size = ng_function->get_graph_size() / 1024;  // kb unit

    // Serialize to nGraph if needed
    if (std::getenv("NGRAPH_ENABLE_SERIALIZE") != nullptr) {
      std::string file_name = "tf_function_" + ctx_params.first + ".json";
      NgraphSerialize("tf_function_" + ctx_params.first + ".json", ng_function);
#if defined NGRAPH_DISTRIBUTED
      int rank_id;
      rank_id = ng::get_distributed_interface()->get_rank();
      NgraphSerialize("tf_function_" + ctx_params.first + "_" +
                          to_string(rank_id) + ".json",
                      ng_function);
#endif
    }
    // Evict the cache if the number of elements exceeds the limit
    const char* cache_depth_specified =
        std::getenv("NGRAPH_TF_FUNCTION_CACHE_ITEM_DEPTH");
    if (cache_depth_specified != nullptr) {
      my_function_cache_depth_in_items = atoi(cache_depth_specified);
    }

    if (m_ng_exec_map.size() >= my_function_cache_depth_in_items) {
      int input_tensors_bytes_free = 0;
      evicted_ng_exec = m_ng_exec_map[m_lru.back()];
      m_ng_exec_map.erase(m_lru.back());
      m_ng_function_map.erase(evicted_ng_exec);

      // Call delete function here pf he erased func
      op_backend->remove_compiled_function(evicted_ng_exec);

      // Now clean the input cache
      std::vector<std::pair<void*, std::shared_ptr<ng::runtime::Tensor>>>&
          input_caches = m_ng_exec_input_cache_map[evicted_ng_exec];
      for (auto& next_input : input_caches) {
        input_tensors_bytes_free += next_input.second->get_size_in_bytes();
        next_input.second.reset();
      }
      m_ng_exec_input_cache_map.erase(evicted_ng_exec);

      // Clean the output cache
      std::vector<std::pair<void*, std::shared_ptr<ng::runtime::Tensor>>>&
          output_caches = m_ng_exec_output_cache_map[evicted_ng_exec];
      int output_tensors_bytes_free = 0;
      for (auto& next_output : output_caches) {
        output_tensors_bytes_free += next_output.second->get_size_in_bytes();
        next_output.second.reset();
      }
      m_ng_exec_output_cache_map.erase(evicted_ng_exec);
      m_lru.pop_back();
      NGRAPH_VLOG(1) << "NGRAPH_TF_MEM_PROFILE:  OP_ID: " << my_instance_id
                     << " Step_ID: " << ctx_params.second
                     << " Cluster: " << ctx_params.first
                     << " Input Tensors freed: "
                     << input_tensors_bytes_free / (1024 * 1024) << " MB"
                     << " Output Tensors freed: "
                     << output_tensors_bytes_free / (1024 * 1024) << " MB";
    }  // cache eviction if cache size greater than cache depth

    BackendManager::LockBackend(m_op_backend_name);

    ngraph::Event event_compile("Compile nGraph", m_name, "");
    try {
      ng_exec = op_backend->compile(ng_function);

    } catch (const std::exception& exp) {
      BackendManager::UnlockBackend(m_op_backend_name);
      NgraphSerialize("tf_function_error_" + ctx_params.first + ".json",
                      ng_function);
      return errors::Internal("Caught exception while compiling op_backend: ",
                              exp.what(), "\n");
    } catch (...) {
      BackendManager::UnlockBackend(m_op_backend_name);
      NgraphSerialize("tf_function_error_" + ctx_params.first + ".json",
                      ng_function);
      return errors::Internal("Error in compiling op_backend\n");
    }
    BackendManager::UnlockBackend(m_op_backend_name);
    event_compile.Stop();
    ngraph::Event::write_trace(event_compile);

    m_ng_exec_map[signature] = ng_exec;
    // caching ng_function to serialize to ngraph if needed
    m_ng_function_map[ng_exec] = ng_function;

    m_lru.push_front(signature);
    // Memory after
    MemoryProfile(vm, rss);
    auto delta_vm_mem = vm - vm0;
    auto delta_res_mem = rss - rss0;
    NGRAPH_VLOG(1) << "NGRAPH_TF_CACHE_PROFILE: OP_ID: " << my_instance_id
                   << " Step_ID: " << ctx_params.second
                   << " Cache length: " << m_ng_exec_map.size()
                   << "  Cluster: " << ctx_params.first
                   << " Delta VM: " << delta_vm_mem
                   << "  Delta RSS: " << delta_res_mem
                   << "  Function size: " << function_size
                   << " KB Total RSS: " << rss / (1024 * 1024) << " GB "
                   << " VM: " << vm / (1024 * 1024) << " GB" << endl;
  }  // end of input signature not found in m_ng_exec_map
  else {
    // Found the input signature in m_ng_exec_map, use the cached executable
    // Update the m_lru
    if (signature != m_lru.front()) {
      m_lru.remove(signature);
      m_lru.push_front(signature);
    }
    ng_exec = it->second;
  }
  return Status::OK();
}

Status NGraphEncapsulateImpl::AllocateNGInputTensors(
    const std::vector<Tensor>& input_tensors,
    std::shared_ptr<ngraph::runtime::Executable>& ng_exec,
    std::vector<TensorShape>& input_shapes, ng::runtime::Backend* op_backend,
    vector<shared_ptr<ng::runtime::Tensor>>& ng_inputs) {
  std::vector<std::unique_ptr<ngraph::Event>> input_copy_events;

  std::vector<std::pair<void*, std::shared_ptr<ng::runtime::Tensor>>>&
      input_caches = m_ng_exec_input_cache_map[ng_exec];
  input_caches.resize(input_shapes.size());
#if defined(NGRAPH_TF_ENABLE_VARIABLES_AND_OPTIMIZERS)
  bool log_copies = false;
  TF_RETURN_IF_ERROR(IsNgraphTFLogTensorCopiesEnabled(m_graph_id, log_copies));
  std::stringstream copy_log_str;
  copy_log_str << "KERNEL[" << type_string() << "]: " << name() << " ,GraphID "
               << m_graph_id << "\n";
  int number_of_copies = 0;
#endif

  for (int i = 0; i < input_shapes.size(); i++) {
#if defined(NGRAPH_TF_ENABLE_VARIABLES_AND_OPTIMIZERS)
    bool ref_exists = NGraphCatalog::ExistsInInputVariableSharedNameMap(
        m_graph_id, def().name(), i);

    // If the input is from a Variable node, we are dealing with later
    // just add a nullptr to the ng_inputs vector.
    if (ref_exists) {
      NGRAPH_VLOG(4) << "NGraphEncapsulateOp:: Input from Variable Node";
      ng_inputs.push_back(nullptr);
      continue;
    }
    NGRAPH_VLOG(4) << "NGraphEncapsulateOp:: Input from non Variable Node";
#endif
    ng::Shape ng_shape(input_shapes[i].dims());
    for (int j = 0; j < input_shapes[i].dims(); ++j) {
      ng_shape[j] = input_shapes[i].dim_size(j);
    }
    ng::element::Type ng_element_type;
    TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(input_tensors[i].dtype(),
                                                     &ng_element_type));

    // At the first call of the ng_exec, both last_src_ptr and
    // last_ng_tensor shall point to null. Otherwise, they are retrived
    // from cache.
    void* last_src_ptr = input_caches[i].first;
    std::shared_ptr<ng::runtime::Tensor> last_ng_tensor =
        input_caches[i].second;
    void* current_src_ptr = (void*)DMAHelper::base(&input_tensors[i]);
    std::shared_ptr<ng::runtime::Tensor> current_ng_tensor =
        GetCurrentNgTensor(current_src_ptr, last_src_ptr, last_ng_tensor, false,
                           ng_exec, op_backend, ng_element_type, ng_shape);
    bool is_cpu = m_op_backend_name == "CPU";

    if (!is_cpu && current_ng_tensor->get_stale()) {
      // Fresh or stale, in case of CPU this step is never needed
      try {
#if defined(NGRAPH_TF_ENABLE_VARIABLES_AND_OPTIMIZERS)
        number_of_copies++;
        copy_log_str << " COPY_INP_VAL[" << i << "]";
#endif
        size_t copy_size =
            current_ng_tensor->get_element_count() * ng_element_type.size();
        string event_name =
            "Input_" + to_string(i) + "_" + to_string(copy_size);
        std::unique_ptr<ngraph::Event> event_copy_input_next(
            new ngraph::Event(event_name, m_name, ""));
        current_ng_tensor->write(
            current_src_ptr, 0,
            current_ng_tensor->get_element_count() * ng_element_type.size());

        event_copy_input_next->Stop();
        input_copy_events.push_back(std::move(event_copy_input_next));

      } catch (const std::exception& exp) {
        errors::Internal(
            "Caught exception while transferring tensor data to nGraph\n");
      } catch (...) {
        errors::Internal("Error in transferring tensor data to nGraph\n");
      }
    }
    input_caches[i] = std::make_pair(current_src_ptr, current_ng_tensor);
    ng_inputs.push_back(current_ng_tensor);
  }  // for (int i = 0; i < input_shapes.size(); i++)

  // Now write the events back
  for (auto& next : input_copy_events) {
    ngraph::Event::write_trace(*next.get());
  }
  return Status::OK();
}

Status NGraphEncapsulateImpl::AllocateNGOutputTensors(
    std::vector<Tensor*>& output_tensors,
    std::vector<ng::element::Type> expected_output_types,
    const std::shared_ptr<ngraph::runtime::Executable>& ng_exec,
    std::vector<TensorShape>& input_shapes, ng::runtime::Backend* op_backend,
    vector<shared_ptr<ng::runtime::Tensor>>& ng_outputs,
    std::vector<std::pair<void*, std::shared_ptr<ng::runtime::Tensor>>>&
        output_caches) {
  output_caches.resize(ng_exec->get_results().size());
  // ngraph executable returns get_results, using that to get the tensor shape
  // and element type.
  for (auto i = 0; i < ng_exec->get_results().size(); i++) {
    auto ng_element = ng_exec->get_results()[i];
    auto ng_shape = ng_element->get_shape();
    auto ng_element_type = ng_element->get_element_type();

    if (ng_element_type != expected_output_types[i]) {
      errors::Internal(
          "Element type inferred by nGraph does not match "
          "the element type expected by TensorFlow");
    }
    void* last_dst_ptr = output_caches[i].first;
    std::shared_ptr<ng::runtime::Tensor> last_ng_tensor =
        output_caches[i].second;

    void* current_dst_ptr = DMAHelper::base(output_tensors[i]);
    std::shared_ptr<ng::runtime::Tensor> current_ng_tensor =
        GetCurrentNgTensor(current_dst_ptr, last_dst_ptr, last_ng_tensor, true,
                           ng_exec, op_backend, ng_element_type, ng_shape);

    current_ng_tensor->set_stale(true);
    output_caches[i] = std::make_pair(current_dst_ptr, current_ng_tensor);
    ng_outputs.push_back(current_ng_tensor);
  }

  return Status::OK();
}

}  // namespace ngraph_bridge

}  // namespace tensorflow