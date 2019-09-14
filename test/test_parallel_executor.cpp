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

TEST(parallel_executor, compiler_test) {
  string graph_name = "test_axpy.pbtxt";
  tensorflow::GraphDef graph_def;
  auto load_graph_status =
      ReadTextProto(Env::Default(), graph_name, &graph_def);
  ASSERT_FALSE(!load_graph_status.ok()) << "Failed to load compute graph";

  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  unique_ptr<tf::Graph> input_graph =
      unique_ptr<tf::Graph>(new tf::Graph(OpRegistry::Global()));
  auto status = ConvertGraphDefToGraph(opts, graph_def, input_graph.get());

  NGraphExecutor executor(100, input_graph);

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
  // tf::ngraph_bridge::BackendManager::SetBackendName("INTERPRETER");

  executor.SetOpBackend("INTERPRETER");
  tf::ngraph_bridge::BackendManager::CreateBackend("INTERPRETER");

  int size = 5;
  executor.ResizeStaticInputVector(size);

  for (int i = 0; i < size; i++) {
    executor.SetStaticInputVector(i, false);
  }

  status = executor.GetNgExecutable(tf_input_tensors, input_shapes,
                                    static_input_map, op_backend, ng_exec);
  ASSERT_EQ(tensorflow::Status::OK(), status);
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow
