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

REGISTER_OP("NGraphEncapsulate")
    .Input("args: Targuments")
    .Attr("Targuments: list(type) >= 0")
    .Output("results: Tresults")
    .Attr("Tresults: list(type) >= 0")
    .Attr("ngraph_cluster: int")
    .Attr("ngraph_graph_id: int")
    .Attr("ngraph_backend: string")
    .SetIsStateful()
    .Doc("nGraph Encapsulation Op. For use by the nGraph JIT only.");

//---------------------------------------------------------------------------
//  NGraphEncapsulateOp::ctor
//---------------------------------------------------------------------------
NGraphEncapsulateOp::NGraphEncapsulateOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) {
  ng_encap_impl = new NGraphEncapsulateImpl(name());

  std::ostringstream oss;
  oss << "Encapsulate_" << ng_encap_impl->get_instance_id() << ": " << name();
  ngraph::Event event(oss.str(), name(), "");

  NGRAPH_VLOG(1) << "NGraphEncapsulateOp: " << ng_encap_impl->get_instance_id()
                 << " Name: " << name();

  GraphDef* graph_def;

  int cluster{-1};
  OP_REQUIRES_OK(ctx, ctx->GetAttr<int>("ngraph_cluster", &cluster));
  ng_encap_impl->set_ngraph_cluster(cluster);

  graph_def = NGraphClusterManager::GetClusterGraph(
      ng_encap_impl->get_ngraph_cluster());

  if (graph_def == nullptr) {
    string flib_key =
        "ngraph_cluster_" + to_string(ng_encap_impl->get_ngraph_cluster());
    // Read graphdef from function library
    const FunctionLibraryDefinition flib =
        *ctx->function_library()->GetFunctionLibraryDefinition();
    const FunctionDef* fdef = flib.Find(flib_key);
    OP_REQUIRES(
        ctx, fdef != nullptr,
        errors::Internal("Did not find graphdef for encapsulate ", flib_key,
                         " in NGraphClusterManager or function library"));
    // TODO: how to convert from functiondef to graphdef. Anything easier?
    std::unique_ptr<FunctionBody> fnbody;
    const auto get_func_sig = [&flib](const string& op, const OpDef** sig) {
      return flib.LookUpOpDef(op, sig);
    };
    FunctionDefToBodyHelper(*fdef, {}, &flib, get_func_sig, &fnbody);
    CopyGraph(*fnbody->graph, &ng_encap_impl->m_graph);
  } else {
    GraphConstructorOptions opts;
    opts.allow_internal_ops = true;
    OP_REQUIRES_OK(
        ctx, ConvertGraphDefToGraph(opts, *graph_def, &ng_encap_impl->m_graph));
  }

  int graph_id{-1};
  OP_REQUIRES_OK(ctx, ctx->GetAttr("ngraph_graph_id", &graph_id));
  ng_encap_impl->set_graph_id(graph_id);
  //
  // Initialize the "m_input_is_static" vector as follows:
  // (1) create m_input_is_static with n+1 elements, where n is the max arg
  //     index
  // (2) for each _Arg node n, set m_input_is_static[n.index] to true if n
  //     is driving any static input; else set it to false.
  //

  // Create the vector.
  int32 max_arg_index = -1;
  std::vector<const Node*> arg_nodes;

  for (auto node : ng_encap_impl->m_graph.nodes()) {
    if (node->type_string() == "_Arg") {
      arg_nodes.push_back(node);

      int32 index;
      OP_REQUIRES_OK(ctx, GetNodeAttr(node->attrs(), "index", &index));
      if (index > max_arg_index) max_arg_index = index;
    }
  }

  ng_encap_impl->get_static() = std::vector<bool>(max_arg_index + 1, false);

  // Fill the vector.
  for (auto node : arg_nodes) {
    int32 index;
    OP_REQUIRES_OK(ctx, GetNodeAttr(node->attrs(), "index", &index));

    bool is_static = false;
    for (auto edge : node->out_edges()) {
      if (edge->IsControlEdge() || !edge->dst()->IsOp()) {
        continue;
      }

      NGRAPH_VLOG(5) << "For arg " << index << " checking edge "
                     << edge->DebugString();

      if (InputIsStatic(edge->dst(), edge->dst_input())) {
        NGRAPH_VLOG(5) << "Marking edge static: " << edge->DebugString();
        is_static = true;
        break;
      }
    }

    NGRAPH_VLOG(5) << "Marking arg " << index << " is_static: " << is_static;
    ng_encap_impl->get_static()[index] = is_static;
  }

  // Set the backend type for the op
  std::string backend_name;
  OP_REQUIRES_OK(ctx, ctx->GetAttr<string>("ngraph_backend", &backend_name));
  // Get the optional attributes
  std::vector<std::string> additional_attributes =
      BackendManager::GetBackendAdditionalAttributes(backend_name);
  std::unordered_map<std::string, std::string> additional_attribute_map;
  for (size_t i = 0; i < additional_attributes.size(); i++) {
    std::string val;
    // Append _ngraph_ to the additional attributes since they
    // are added as optional attributes with a `ngraph` prefix
    // to the encapsulate node
    std::string attr = "_ngraph_" + additional_attributes[i];
    // If an attribute does not exist, TF will return a non-ok status
    OP_REQUIRES_OK(ctx, ctx->GetAttr<string>(attr, &val));
    additional_attribute_map.insert({additional_attributes[i], val});
  }

  // Concatenate the backend_name:backend_config
  try {
    string be_name = BackendManager::GetBackendCreationString(
        backend_name, additional_attribute_map);
    ng_encap_impl->set_op_backend_name(be_name);
  } catch (const std::exception& exp) {
    OP_REQUIRES_OK(
        ctx, errors::Internal("Caught exception while creating backend string ",
                              exp.what(), "\n"));
  }
  NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Create backend " << def().name();
  BackendManager::CreateBackend(backend_name);
  // SetConfig will be called for each EncapsulateOp
  BackendManager::SetConfig(backend_name, additional_attribute_map);

  event.Stop();
  ngraph::Event::write_trace(event);
}

//---------------------------------------------------------------------------
//  ~NGraphEncapsulateOp()
//---------------------------------------------------------------------------
NGraphEncapsulateOp::~NGraphEncapsulateOp() {
  std::ostringstream oss;
  oss << "Destroy Encapsulate_" << ng_encap_impl->get_instance_id() << ": "
      << name();
  ngraph::Event event(oss.str(), name(), "");
  NGRAPH_VLOG(2) << "~NGraphEncapsulateOp::" << name();
  // If the kernel goes away, we must de-register all of its cached
  // functions
  // from the freshness tracker.
  if (ng_encap_impl->m_freshness_tracker != nullptr) {
    for (auto kv : ng_encap_impl->get_ng_exec_map()) {
      ng_encap_impl->m_freshness_tracker->RemoveUser(kv.second);
    }

    // TODO(amprocte): We should be able to unref the tracker here, but it
    // seems to screw things up in the C++ unit tests.
    // m_freshness_tracker->Unref();
  }

#if defined(NGRAPH_TF_ENABLE_VARIABLES_AND_OPTIMIZERS)
  for (int i = 0; i < ng_encap_impl->get_number_outputs(); i++) {
    string key =
        NGraphCatalog::CreateNodeKey(ng_encap_impl->get_graph_id(), name(), i);
    if (NGraphCatalog::ExistsInEncapOutputTensorMap(key)) {
      NGraphCatalog::DeleteFromEncapOutputTensorMap(key);
      NGRAPH_VLOG(2) << "Deleting from output tensor map " << key;
    }
  }
#endif

  // Release the backend
  NGRAPH_VLOG(2) << "~NGraphEncapsulateOp():: ReleaseBackend";
  BackendManager::ReleaseBackend(ng_encap_impl->get_op_backend_name());
  event.Stop();
  ngraph::Event::write_trace(event);
}

//---------------------------------------------------------------------------
// OpKernel::Compute
//---------------------------------------------------------------------------
void NGraphEncapsulateOp::Compute(OpKernelContext* ctx) {
  std::ostringstream oss;
  oss << "Execute: Encapsulate_" << ng_encap_impl->get_instance_id() << ": "
      << name();
  ngraph::Event event(oss.str(), name(), "");

  Timer compute_time;
  std::lock_guard<std::mutex> lock(ng_encap_impl->m_compute_lock);
  NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute starting for cluster "
                 << ng_encap_impl->get_ngraph_cluster();

  ngraph::Event event_func_maybe_create("FunctionMaybeCreate", name(), "");
  Timer function_lookup_or_create;

  std::vector<TensorShape> input_shapes;
  std::vector<const Tensor*> static_input_map;
  std::shared_ptr<ngraph::Function> ng_function;
  std::shared_ptr<ngraph::runtime::Executable> ng_exec;
  ng::runtime::Backend* op_backend;

  // TF input tensor
  std::vector<Tensor> tf_input_tensors;

  for (int i = 0; i < ctx->num_inputs(); i++) {
    tf_input_tensors.push_back(ctx->input(i));
  }

  std::pair<string, int64> ctx_params;
  ctx_params.first = ctx->op_kernel().name();
  ctx_params.second = ctx->step_id();

  // Get ngraph executable and inputs information
  OP_REQUIRES_OK(ctx, ng_encap_impl->GetNgExecutable(
                          tf_input_tensors, ctx_params, input_shapes,
                          static_input_map, op_backend, ng_exec));

  NGRAPH_VLOG(4)
      << "NGraphEncapsulateOp::Compute got ngraph executable for cluster "
      << ng_encap_impl->get_ngraph_cluster();

  int time_func_create_or_lookup = function_lookup_or_create.ElapsedInMS();
  event_func_maybe_create.Stop();

  NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute got graph for cluster "
                 << ng_encap_impl->get_ngraph_cluster();

  Timer create_or_lookup_tensors;

  if (ng_encap_impl->m_freshness_tracker == nullptr) {
    auto creator = [](NGraphFreshnessTracker** tracker) {
      *tracker = new NGraphFreshnessTracker();
      return Status::OK();
    };
    OP_REQUIRES_OK(
        ctx, ctx->resource_manager()->LookupOrCreate<NGraphFreshnessTracker>(
                 ctx->resource_manager()->default_container(),
                 "ngraph_freshness_tracker",
                 &ng_encap_impl->m_freshness_tracker, creator));
  }

  NGRAPH_VLOG(4)
      << "NGraphEncapsulateOp::Compute got freshness tracker for cluster "
      << ng_encap_impl->get_ngraph_cluster();

  // Allocate tensors for input arguments.
  ngraph::Event event_alloc_input("Input: maybe create", name(), "");

  vector<shared_ptr<ng::runtime::Tensor>> ng_inputs;
  int ng_input_tensor_size_in_bytes = 0;

  OP_REQUIRES_OK(
      ctx, ng_encap_impl->AllocateNGInputTensors(
               tf_input_tensors, ng_exec, input_shapes, op_backend, ng_inputs));

  event_alloc_input.Stop();

  NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute allocated argument tensors "
                    "for cluster "
                 << ng_encap_impl->get_ngraph_cluster();

  // Allocate tensors for the output results.
  ngraph::Event event_alloc_output("Output: maybe create", name(), "");
  vector<shared_ptr<ng::runtime::Tensor>> ng_outputs;
  int ng_output_tensor_size_in_bytes = 0;
  std::vector<std::pair<void*, std::shared_ptr<ng::runtime::Tensor>>>&
      output_caches = ng_encap_impl->get_ng_exec_output_cache_map()[ng_exec];

  std::vector<Tensor*> tf_output_tensors;
  std::vector<ng::element::Type> expected_output_types;
  for (auto i = 0; i < ng_exec->get_results().size(); i++) {
    auto ng_element = ng_exec->get_results()[i];
    auto ng_shape = ng_element->get_shape();
    auto ng_element_type = ng_element->get_element_type();

    // Create the TF output tensor
    vector<int64> dims;
    for (auto dim : ng_shape) {
      dims.push_back(dim);
    }
    TensorShape tf_shape(dims);
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(i, tf_shape, &output_tensor));
    tf_output_tensors.push_back(output_tensor);

    // Make sure the nGraph-inferred element type agrees with what TensorFlow
    // expected.
    ng::element::Type expected_elem_type;
    OP_REQUIRES_OK(ctx,
                   TFDataTypeToNGraphElementType(ctx->expected_output_dtype(i),
                                                 &expected_elem_type));
    expected_output_types.push_back(expected_elem_type);
  }

  OP_REQUIRES_OK(ctx, ng_encap_impl->AllocateNGOutputTensors(
                          tf_output_tensors, expected_output_types, ng_exec,
                          input_shapes, op_backend, ng_outputs, output_caches));

  event_alloc_output.Stop();

  NGRAPH_VLOG(4)
      << "NGraphEncapsulateOp::Compute allocated result tensors for cluster "
      << ng_encap_impl->get_ngraph_cluster();

#if defined(NGRAPH_TF_ENABLE_VARIABLES_AND_OPTIMIZERS)
  NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute getting input variables "
                    "from resource manager "
                 << ng_encap_impl->get_ngraph_cluster();

  ngraph::Event event_input_check_in_catalog(
      "Get Variable Inputs from Resource Manager", name(), "");

  // Dealing with the input from Variable nodes here
  for (int input_index = 0; input_index < input_shapes.size(); input_index++) {
    bool ref_exists = NGraphCatalog::ExistsInInputVariableSharedNameMap(
        ng_encap_impl->get_graph_id(), def().name(), input_index);

    if (!ref_exists) {
      OP_REQUIRES(ctx, ng_inputs[input_index] != nullptr,
                  errors::Internal("Input ", input_index,
                                   " is not in Catalog nor was set from TF"));
      continue;
    }

    string ref_var_name = NGraphCatalog::GetInputVariableSharedName(
        ng_encap_impl->get_graph_id(), def().name(), input_index);
    NGraphVar* var;
    OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup<NGraphVar>(
                            ctx->resource_manager()->default_container(),
                            ref_var_name, &var));

    if (var->sync_ng_tensor()) {
      int copies = ng_encap_impl->get_number_of_copies();
      ng_encap_impl->set_number_of_copies(copies++);
      ng_encap_impl->get_log_copies() << "Var_Sync[" << input_index << "] ";
    }

    void* current_tf_ptr = (void*)DMAHelper::base(&ctx->input(input_index));
    bool is_stale =
        !ng_encap_impl->m_freshness_tracker->IsFresh(current_tf_ptr, ng_exec);
    var->ng_tensor()->set_stale(is_stale);
    ng_inputs[input_index] = var->ng_tensor();

    var->Unref();
  }

  event_input_check_in_catalog.Stop();
  ngraph::Event::write_trace(event_input_check_in_catalog);
#endif

  int time_create_or_lookup_tensors = create_or_lookup_tensors.ElapsedInMS();

  // Execute the nGraph function.
  ngraph::Event event_execute_function("Execute nGraph", name(), "");
  Timer execute_function;
  {
    BackendManager::LockBackend(ng_encap_impl->get_op_backend_name());
    NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute call starting for cluster "
                   << ng_encap_impl->get_ngraph_cluster();
    try {
      ng_exec->call(ng_outputs, ng_inputs);
    } catch (const std::exception& exp) {
      ng_function = ng_encap_impl->get_ng_function_map()[ng_exec];
      BackendManager::UnlockBackend(ng_encap_impl->get_op_backend_name());
      NgraphSerialize("tf_function_error_" + ctx->op_kernel().name() + ".json",
                      ng_function);
      OP_REQUIRES(ctx, false,
                  errors::Internal(
                      "Caught exception while executing nGraph computation: ",
                      exp.what(), "\n"));
    } catch (...) {
      ng_function = ng_encap_impl->get_ng_function_map()[ng_exec];
      BackendManager::UnlockBackend(ng_encap_impl->get_op_backend_name());
      NgraphSerialize("tf_function_error_" + ctx->op_kernel().name() + ".json",
                      ng_function);
      OP_REQUIRES(
          ctx, false,
          errors::Internal("Error in executing the nGraph computation\n"));
    }
    BackendManager::UnlockBackend(ng_encap_impl->get_op_backend_name());
  }
  int time_execute_function = execute_function.ElapsedInMS();
  event_execute_function.Stop();

  long vm, rss;
  MemoryProfile(vm, rss);
  NGRAPH_VLOG(1) << "NGRAPH_TF_MEM_PROFILE:  OP_ID: "
                 << ng_encap_impl->get_instance_id()
                 << " Step_ID: " << ctx_params.second
                 << " Cluster: " << ctx_params.first
                 << " Input Tensors created: "
                 << ng_input_tensor_size_in_bytes / (1024 * 1024) << " MB"
                 << " Output Tensors created: "
                 << ng_output_tensor_size_in_bytes / (1024 * 1024) << " MB"
                 << " Total process memory: " << rss / (1024 * 1024) << " GB";

  NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute call done for cluster "
                 << ng_encap_impl->get_ngraph_cluster();

  // Copy value to host if backend is not CPU
  ngraph::Event event_copy_output("Output - copy back", name(), "");
  Timer copy_output_tensors_to_host;

  try {
    size_t output_tensor_count = output_caches.size();
    std::vector<std::unique_ptr<ngraph::Event>> output_copy_events;
#if defined(NGRAPH_TF_ENABLE_VARIABLES_AND_OPTIMIZERS)
    if (ng_encap_impl->get_number_outputs() == -1) {
      NGRAPH_VLOG(4) << "Settig number of outputs for " << def().name();
      ng_encap_impl->get_number_outputs() = output_caches.size();
    }
    for (size_t i = 0; i < output_tensor_count; ++i) {
      string key = NGraphCatalog::CreateNodeKey(ng_encap_impl->get_graph_id(),
                                                def().name(), i);
      bool ref_exists = NGraphCatalog::ExistsInEncapOutputTensorMap(key);
      void* dst_ptr;
      std::shared_ptr<ng::runtime::Tensor> dst_ng_tensor;
      std::tie(dst_ptr, dst_ng_tensor) = output_caches[i];

      if (ref_exists) {
        NGRAPH_VLOG(4) << "Adding in output tensor map " << key;
        NGraphCatalog::AddToEncapOutputTensorMap(key, dst_ng_tensor);
      }

      if (ng_encap_impl->get_op_backend_name() != "CPU" &&
          NGraphCatalog::EncapOutputIndexNeedsCopy(def().name(), i)) {
        int copies = ng_encap_impl->get_number_of_copies();
        ng_encap_impl->set_number_of_copies(copies++);
        ng_encap_impl->get_log_copies() << " COPY_OP_VAL[" << i << "]";

        NGRAPH_VLOG(4) << "Copying Output " << def().name() << " ,index: " << i;
        auto ng_element_type = dst_ng_tensor->get_element_type();
        size_t copy_size =
            dst_ng_tensor->get_element_count() * ng_element_type.size();
        string event_name =
            "Output_" + to_string(i) + "_" + to_string(copy_size);
        std::unique_ptr<ngraph::Event> event_copy_output_next(
            new ngraph::Event(event_name, name(), ""));
        dst_ng_tensor->read(dst_ptr, 0, dst_ng_tensor->get_element_count() *
                                            ng_element_type.size());
        event_copy_output_next->Stop();
        output_copy_events.push_back(std::move(event_copy_output_next));
      }
    }
#else
    if (ng_encap_impl->get_op_backend_name() != "CPU") {
      for (size_t i = 0; i < output_tensor_count; ++i) {
        void* dst_ptr;
        std::shared_ptr<ng::runtime::Tensor> dst_ng_tensor;
        std::tie(dst_ptr, dst_ng_tensor) = output_caches[i];
        auto ng_element_type = dst_ng_tensor->get_element_type();
        std::unique_ptr<ngraph::Event> event_copy_output_next(new ngraph::Event(
            ("Output_" + std::to_string(i) + "_" +
             std::to_string(dst_ng_tensor->get_element_count() *
                            ng_element_type.size())),
            name(), ""));
        dst_ng_tensor->read(dst_ptr, 0, dst_ng_tensor->get_element_count() *
                                            ng_element_type.size());
        event_copy_output_next->Stop();
        output_copy_events.push_back(std::move(event_copy_output_next));
      }
    }
#endif
    // Now write the events back
    for (auto& next : output_copy_events) {
      ngraph::Event::write_trace(*next.get());
    }
  } catch (const std::exception& exp) {
    OP_REQUIRES(ctx, false,
                errors::Internal(
                    "Caught exception while transferring tensor data to host: ",
                    exp.what(), "\n"));
  } catch (...) {
    OP_REQUIRES(ctx, false, errors::Internal(
                                "Error in transferring tensor data to host\n"));
  }
  event_copy_output.Stop();

#if defined(NGRAPH_TF_ENABLE_VARIABLES_AND_OPTIMIZERS)
  ng_encap_impl->get_log_copies()
      << " Number of copies " << ng_encap_impl->get_number_of_copies() << "\n";
  if (ng_encap_impl->get_log_copies() {
    cout << ng_encap_impl->get_log_copies().str();
  }
#endif

  // Mark input tensors as fresh for the next time around.
  // Note: these ng_tensors are being marked fresh so that in the next
  // iteration if this encapsulate finds the tensor fresh, then it will use it
  for (int i = 0; i < input_shapes.size(); i++) {
    void* src_ptr = (void*)DMAHelper::base(&ctx->input(i));
    ng_encap_impl->m_freshness_tracker->MarkFresh(src_ptr, ng_exec);
  }
  int time_copy_output_tensors_to_host =
      copy_output_tensors_to_host.ElapsedInMS();

  NGRAPH_VLOG(4)
      << "NGraphEncapsulateOp::Compute done marking fresh for cluster "
      << ng_encap_impl->get_ngraph_cluster();
  NGRAPH_VLOG(1)
      << "NGRAPH_TF_TIMING_PROFILE: OP_ID: " << ng_encap_impl->get_instance_id()
      << " Step_ID: " << ctx_params.second << " Cluster: " << ctx_params.first
      << " Time-Compute: " << compute_time.ElapsedInMS()
      << " Function-Create-or-Lookup: " << time_func_create_or_lookup
      << " Create-and-copy-tensors: " << time_create_or_lookup_tensors
      << " Execute: " << time_execute_function
      << " Copy-outputs-to-host: " << time_copy_output_tensors_to_host;
  event.Stop();
  ngraph::Event::write_trace(event_func_maybe_create);
  ngraph::Event::write_trace(event_alloc_output);
  ngraph::Event::write_trace(event_alloc_input);
  ngraph::Event::write_trace(event_execute_function);
  ngraph::Event::write_trace(event_copy_output);
  ngraph::Event::write_trace(event);

}  // end compute

int NGraphEncapsulateImpl::s_instance_count = 0;

}  // namespace ngraph_bridge

REGISTER_KERNEL_BUILDER(Name("NGraphEncapsulate").Device(DEVICE_CPU),
                        ngraph_bridge::NGraphEncapsulateOp);

}  // namespace tensorflow