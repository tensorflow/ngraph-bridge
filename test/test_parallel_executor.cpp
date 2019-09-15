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
#include "ngraph_bridge/ngraph_executor.h"
#include "ngraph_bridge/version.h"
#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;
namespace tf = tensorflow;

namespace tensorflow {
namespace ngraph_bridge {
namespace testing {

Status LoadGraphFromPbTxt(const string& pb_file, const string& backend_name,
                          unique_ptr<tf::Graph>& new_graph, unique_ptr<tf::Session>& session) {
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

  // Note: The status returned from the function below will complain that:
  //
  // Op type not registered 'Constant' in binary running on <host-name>. 
  // Make sure the Op and Kernel are registered in the binary running in 
  // this process. Note that if you are loading a saved graph which used ops 
  // from tf.contrib, accessing (e.g.) `tf.contrib.resampler` should be done 
  // before importing the graph, as contrib ops are lazily registered when the 
  // module is first accessed.
  //
  // This is because we haven't loaded the TF op registration modules
  // and is not needed for the graph conversion. So we will ignore 
  // the resulting status.
  // To see the error message of the returned Status:
  // Call: cout << status.error_message() << endl;
  // After the following line

  auto status = ConvertGraphDefToGraph(opts, graph_def, input_graph.get());
  new_graph = move(input_graph);
  return Status::OK();
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
  unique_ptr<tf::Graph> input_graph;
  unique_ptr<tf::Session> session;

  ASSERT_OK(LoadGraphFromPbTxt("test_axpy_const.pbtxt", "INTERPRETER", input_graph, session));

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

  // TODO: Investigate is the executor can decipher the static inputs
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
  ASSERT_OK(status);
  ASSERT_FALSE(cache_hit);

  // Now call again to test that the cache works
  ASSERT_OK(executor.GetNgExecutable(tf_input_tensors, input_shapes, static_input_map,
                            op_backend, ng_exec, cache_hit));
  
  // If the cache doesn't work then the following will fire
  ASSERT_TRUE(cache_hit);

  // Now validate that the nGraph function is available
  std::shared_ptr<ngraph::Function> ng_function;
  ASSERT_EQ(executor.GetNgFunction(ng_exec, ng_function),
            tensorflow::Status::OK());

  // Validate the nGraph Function
  const auto& parameters = ng_function->get_parameters();

  cout << " Friendly name: " << ng_function->get_friendly_name()
       << " PArameters: " << parameters.size() << std::endl;
}

TEST(parallel_executor, DISABLED_execute_one_thread) {
  // Read the graph
  unique_ptr<tf::Graph> input_graph;
  unique_ptr<tf::Session> session;
  LoadGraphFromPbTxt("test_axpy_const.pbtxt", "INTERPRETER", input_graph, session);
  // Note - we are NOT checking the return status as the status will be non OK
  // due to no TF Op registration being done yet. For our test - we don't need
  // to
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

  // TODO: Investigate is the executor can decipher the static inputs
  // from the given graph (as opposed to feeding this in externally)
  int size = 5;
  executor.ResizeStaticInputVector(size);

  for (int i = 0; i < size; i++) {
    executor.SetStaticInputVector(i, false);
  }

  bool cache_hit = false;
  ASSERT_OK(executor.GetNgExecutable(tf_input_tensors, input_shapes, static_input_map,
                               op_backend, ng_exec, cache_hit));
  ASSERT_FALSE(cache_hit);

  int pipeline_idx = -1;
  PipelinedTensorVector inp_group_from_pipeline;
  PipelinedTensorVector out_group_from_pipeline;

  executor.CachePipelinedTensorIfNeeded(ng_exec);
  std::cout << "HERE" << std::endl;

  // Get the pipelned tensors
  ASSERT_NO_THROW(
      std::tie(pipeline_idx, inp_group_from_pipeline, out_group_from_pipeline) =
          executor.GetTensorsFromPipeline(ng_exec));

  ASSERT_GE(pipeline_idx, 0) << "GetTensorsFromPipeline() Returned: "
                             << pipeline_idx;
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow
