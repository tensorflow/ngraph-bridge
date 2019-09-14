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

#include "ngraph_bridge/ngraph_backend_manager.h"
#include "ngraph_bridge/ngraph_executor.h"

using namespace std;
namespace ng = ngraph;
namespace tf = tensorflow;

namespace tensorflow {
namespace ngraph_bridge {
namespace testing {

Status LoadGraphFromPbTxt(const string& pb_file, unique_ptr<tf::Graph>& new_graph){
  // Read the graph
  tensorflow::GraphDef graph_def;
  auto load_graph_status =
      ReadTextProto(Env::Default(), pb_file, &graph_def);
  if (!load_graph_status.ok()) {
    return errors::Internal("Failed to load compute graph");
  }

  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  unique_ptr<tf::Graph> input_graph =
      unique_ptr<tf::Graph>(new tf::Graph(OpRegistry::Global()));
  auto status = ConvertGraphDefToGraph(opts, graph_def, input_graph.get());
  new_graph = move(input_graph);
  return status;
}

TEST(parallel_executor, construction) {
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  unique_ptr<tf::Graph> input_graph =
      unique_ptr<tf::Graph>(new tf::Graph(OpRegistry::Global()));

  // First test with a backend not yet created
  unique_ptr<NGraphExecutor> executor;
  ASSERT_THROW(executor = unique_ptr<NGraphExecutor>(
                   new NGraphExecutor(100, input_graph, "bogus")),
               std::runtime_error);

  // Next test with a backend after creating
  tf::ngraph_bridge::BackendManager::CreateBackend("INTERPRETER");
  ASSERT_NO_THROW(executor = unique_ptr<NGraphExecutor>(
                      new NGraphExecutor(100, input_graph, "INTERPRETER")));

  // Now that the object has been cobstructed, test various internal parts
  // TODO: Create a Test Class and mark that as a friend of the Executor class
  ASSERT_EQ(executor->GetOpBackend(), "INTERPRETER");
  ASSERT_TRUE(executor->GetExecCanCreateTensor());
}

TEST(parallel_executor, compiler_test) {
  // TODO: Need to use a more realistic graph with _Arg and _Retval
  // addded i.e., a PB that is saved after the initial processing of the
  // TF graph transformation.
  // Call Grappler here to get the graph transformed?

  // Read the graph
  string graph_name = "test_axpy_const.pbtxt";

  unique_ptr<tf::Graph> input_graph; 
  LoadGraphFromPbTxt(graph_name, input_graph);
  // Note - we are NOT checking the return status as the status will be non OK
  // due to no TF Op registration tdone yet. For out test - we don't need to 
  // worry about it as the TF Ops will be converted to nGraph Op

  tf::ngraph_bridge::BackendManager::CreateBackend("INTERPRETER");
  NGraphExecutor executor(100, input_graph, "INTERPRETER");

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
  vector<TensorShape> input_shapes;
  vector<const Tensor*> static_input_map;
  ng::runtime::Backend* op_backend;
  shared_ptr<ngraph::runtime::Executable> ng_exec;

  // Call the Executor to compile the funcion
  // executor.SetOpBackend("INTERPRETER");

  // TODO: Investigate is the executor can decifer the static inputs
  // from the given graph (as opposed to feeding this in externally)
  int size = 5;
  executor.ResizeStaticInputVector(size);

  for (int i = 0; i < size; i++) {
    executor.SetStaticInputVector(i, false);
  }

  bool cache_hit = false;
  Status status =
      executor.GetNgExecutable(tf_input_tensors, input_shapes, static_input_map,
                               op_backend, ng_exec, cache_hit);
  ASSERT_EQ(tensorflow::Status::OK(), status);
  ASSERT_FALSE(cache_hit);

  // Now call again to test that the cache works
  status =
      executor.GetNgExecutable(tf_input_tensors, input_shapes, static_input_map,
                               op_backend, ng_exec, cache_hit);
  ASSERT_EQ(tensorflow::Status::OK(), status);
  // If the cache doesn't work then the following will fire
  ASSERT_TRUE(cache_hit);
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow
