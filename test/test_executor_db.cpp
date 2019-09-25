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
#include "test/test_generic_class.h"
#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;
namespace tf = tensorflow;

namespace tensorflow {
namespace ngraph_bridge {
namespace testing {

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

class NGraphExecutorDBTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tf::ngraph_bridge::testing::GenericUtil::LoadGraphFromPbTxt(
        "test_axpy_launchop.pbtxt", input_graph);
    tf::ngraph_bridge::BackendManager::CreateBackend("INTERPRETER");
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
    tf_input_tensors.push_back(x);
    tf_input_tensors.push_back(y);
    ComputeSignature(tf_input_tensors, input_shapes, signature_ss);
    signature = signature_ss.str();
  }

  // void TearDown() override { }
  unique_ptr<tf::Graph> input_graph;
  NGraphExecutorDB edb;
  std::vector<Tensor> tf_input_tensors;
  std::vector<TensorShape> input_shapes;
  shared_ptr<ngraph::runtime::Executable> ng_exec;
  std::stringstream signature_ss;
  std::string signature;
  std::shared_ptr<ngraph::Function> ng_function;
};

TEST_F(NGraphExecutorDBTest, CompileExe) {
  ASSERT_EQ(edb.IsNgExecAvail(signature, ng_exec), false);
  ASSERT_OK(CompileExecutable(edb, input_shapes, input_graph.get(), ng_function,
                              ng_exec, signature));

  ASSERT_EQ(edb.IsNgFunctionAvail(ng_exec, ng_function), true);

  // Validate the nGraph Function
  const auto& parameters = ng_function->get_parameters();
  ASSERT_EQ(2, parameters.size());
}
}
}
}