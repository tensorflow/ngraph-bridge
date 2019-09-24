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
#include "gtest/gtest.h"

#include <memory>
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/public/session.h"

#include "ngraph_bridge/ngraph_backend_manager.h"
#include "ngraph_bridge/ngraph_builder.h"
#include "ngraph_bridge/ngraph_executor_db.h"
#include "ngraph_bridge/version.h"
#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;
namespace tf = tensorflow;

namespace tensorflow {
namespace ngraph_bridge {
namespace testing {

Status ExLoadGraphFromPbTxt(const string& pb_file, const string& backend_name,
                            unique_ptr<tf::Graph>& new_graph) {
  // Read the graph
  tensorflow::GraphDef graph_def;
  auto load_graph_status = ReadTextProto(Env::Default(), pb_file, &graph_def);
  if (!load_graph_status.ok()) {
    return load_graph_status;
  }

  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  unique_ptr<tf::Graph> input_graph =
      unique_ptr<tf::Graph>(new tf::Graph(OpRegistry::Global()));

  auto status = ConvertGraphDefToGraph(opts, graph_def, input_graph.get());
  new_graph = move(input_graph);
  return status;
}

Status ComputeSignature(const std::vector<Tensor>& tf_input_tensors,
                        std::vector<TensorShape>& input_shapes,
                        std::stringstream& signature_ss) {
  // Use tensorflow input tensors to get input_shapes, static_input_map
  // and compute the signature
  for (int i = 0; i < tf_input_tensors.size(); i++) {
    const Tensor& input_tensor = tf_input_tensors[i];
    input_shapes.push_back(input_tensor.shape());
    for (const auto& x : input_tensor.shape()) {
      signature_ss << x.size << ",";
    }
    signature_ss << ";";
  }
  return Status::OK();
}

Status CompileExecutable(NGraphExecutorDB& edb,
                         std::vector<TensorShape> input_shapes,
                         tf::Graph* graph,
                         std::shared_ptr<ngraph::Function>& ng_function,
                         shared_ptr<ngraph::runtime::Executable>& ng_exec,
                         std::string signature) {
  std::vector<const Tensor*> static_input_map;
  std::shared_ptr<ngraph::runtime::Executable> evicted_ng_exec;
  Builder::TranslateGraph(input_shapes, static_input_map, graph, ng_function);

  ng::runtime::Backend* op_backend;

  op_backend = BackendManager::GetBackend("INTERPRETER");
  const char* cache_depth_specified =
      std::getenv("NGRAPH_TF_FUNCTION_CACHE_ITEM_DEPTH");
  int my_function_cache_depth_in_items = 16;
  if (cache_depth_specified != nullptr) {
    my_function_cache_depth_in_items = atoi(cache_depth_specified);
  }
  if (edb.SizeOfNgExecMap() >= my_function_cache_depth_in_items) {
    edb.RemoveExecAndFunc(evicted_ng_exec);
    op_backend->remove_compiled_function(evicted_ng_exec);
    edb.PopBackLRU();
  }

  ngraph::Event event_compile("Compile nGraph", "", "");
  BackendManager::LockBackend("INTERPRETER");
  try {
    ng_exec = op_backend->compile(ng_function);
  } catch (const std::exception& exp) {
    BackendManager::UnlockBackend("INTERPRETER");
    return errors::Internal(" cccc");
  } catch (...) {
    BackendManager::UnlockBackend("INTERPRETER");
    return errors::Internal(" cccc");
  }
  BackendManager::UnlockBackend("INTERPRETER");
  event_compile.Stop();

  ngraph::Event::write_trace(event_compile);
  edb.InsertNgExecMap(signature, ng_exec);
  edb.InsertNgFunctionMap(ng_exec, ng_function);
  edb.PushFrontInLRU(signature);
  return Status::OK();
}

TEST(ExecutorDB, CompilerTest) {
  // Read the graph
  unique_ptr<tf::Graph> input_graph;

  // We are using a graph with _Arg and _Retval
  // addded i.e., a PB that is saved after the initial processing of the
  // TF graph transformation.
  ASSERT_OK(ExLoadGraphFromPbTxt("test_axpy_launchop.pbtxt", "INTERPRETER",
                                 input_graph));

  tf::ngraph_bridge::BackendManager::CreateBackend("INTERPRETER");
  NGraphExecutorDB edb;
  // Create the inputs for this graph
  Tensor x(DT_FLOAT, TensorShape({2, 3}));
  auto x_flat = x.flat<float>();
  for (int i = 0; i < x_flat.size(); i++) {
    x_flat.data()[i] = 1.0;
  }

  Tensor y(DT_FLOAT, TensorShape({2, 3}));
  auto y_flat = y.flat<float>();
  for (int i = 0; i < y_flat.size(); i++) {
    y_flat.data()[i] = 1.0;
  }

  std::vector<Tensor> tf_input_tensors{x, y};
  shared_ptr<ngraph::runtime::Executable> ng_exec;
  std::vector<TensorShape> input_shapes;
  std::stringstream signature;
  std::shared_ptr<ngraph::Function> ng_function;

  ASSERT_OK(ComputeSignature(tf_input_tensors, input_shapes, signature));

  std::string sig_str = signature.str();
  if (!(edb.IsNgExecAvail(sig_str, ng_exec))) {
    ASSERT_OK(CompileExecutable(edb, input_shapes, input_graph.get(),
                                ng_function, ng_exec, sig_str));
  } else {
    if (sig_str != edb.LRUFront()) {
      edb.RemoveFromLRU(sig_str);
      edb.PushFrontInLRU(sig_str);
    }
  }

  ASSERT_EQ(edb.IsNgFunctionAvail(ng_exec, ng_function), true);

  // Validate the nGraph Function
  const auto& parameters = ng_function->get_parameters();
  ASSERT_EQ(2, parameters.size());
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow
