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
#include <atomic>
#include <memory>
#include "gtest/gtest.h"

#include "ngraph_bridge/ngraph_backend_manager.h"
#include "ngraph_bridge/ngraph_builder.h"
#include "ngraph_bridge/ngraph_executor_db.h"
#include "ngraph_bridge/version.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/public/session.h"
#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;
namespace tf = tensorflow;

namespace tensorflow {
namespace ngraph_bridge {
namespace testing {

#define ASSERT_OK(x) ASSERT_EQ((x), ::tensorflow::Status::OK());
#define ASSERT_NOT_OK(x) ASSERT_NE((x), ::tensorflow::Status::OK());

Status LoadGraphFromPbTxt(const string& pb_file,
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

class NGraphExecutorDBTest : public ::testing::Test {
 private:
  void SetUp() override {
    tf::ngraph_bridge::testing::LoadGraphFromPbTxt("test_axpy_launchop.pbtxt",
                                                   m_input_graph);
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
    m_tf_input_tensors.push_back(x);
    m_tf_input_tensors.push_back(y);
    NGraphExecutorDBTest::ComputeSignature(m_tf_input_tensors, m_input_shapes,
                                           m_signature_ss);
    m_signature = m_signature_ss.str();
  }
  // The line commented below is to indicate that at this point
  // we
  // void TearDown() override { }
 protected:
  unique_ptr<tf::Graph> m_input_graph;
  NGraphExecutorDB m_edb;
  std::vector<Tensor> m_tf_input_tensors;
  std::vector<TensorShape> m_input_shapes;
  shared_ptr<ngraph::runtime::Executable> m_ng_exec;

  std::stringstream m_signature_ss;
  std::string m_signature;
  std::shared_ptr<ngraph::Function> m_ng_function;

  static Status ComputeSignature(const std::vector<Tensor>& m_tf_input_tensors,
                                 std::vector<TensorShape>& m_input_shapes,
                                 std::stringstream& m_signature_ss);

  static Status CompileExecutable(
      NGraphExecutorDB& m_edb, std::vector<TensorShape> m_input_shapes,
      tf::Graph* graph, std::shared_ptr<ngraph::Function>& m_ng_function,
      shared_ptr<ngraph::runtime::Executable>& m_ng_exec,
      std::string m_signature);
};

Status NGraphExecutorDBTest::ComputeSignature(
    const std::vector<Tensor>& m_tf_input_tensors,
    std::vector<TensorShape>& m_input_shapes,
    std::stringstream& m_signature_ss) {
  // Use tensorflow input tensors to get m_input_shapes, static_input_map
  // and compute the m_signature
  for (int i = 0; i < m_tf_input_tensors.size(); i++) {
    const Tensor& input_tensor = m_tf_input_tensors[i];
    m_input_shapes.push_back(input_tensor.shape());
    for (const auto& x : input_tensor.shape()) {
      m_signature_ss << x.size << ",";
    }
    m_signature_ss << ";";
  }
  return Status::OK();
}

Status NGraphExecutorDBTest::CompileExecutable(
    NGraphExecutorDB& m_edb, std::vector<TensorShape> m_input_shapes,
    tf::Graph* graph, std::shared_ptr<ngraph::Function>& m_ng_function,
    shared_ptr<ngraph::runtime::Executable>& m_ng_exec,
    std::string m_signature) {
  std::vector<const Tensor*> static_input_map;
  Builder::TranslateGraph(m_input_shapes, static_input_map, graph,
                          m_ng_function);

  ng::runtime::Backend* op_backend;

  op_backend = BackendManager::GetBackend("INTERPRETER");
  ngraph::Event event_compile("Compile nGraph", "", "");
  BackendManager::LockBackend("INTERPRETER");
  try {
    m_ng_exec = op_backend->compile(m_ng_function);
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
  return Status::OK();
}

TEST_F(NGraphExecutorDBTest, CompileExe) {
  ASSERT_EQ(m_edb.MaybeGetNgExecutable(m_signature, m_ng_exec), false);
  ASSERT_OK(NGraphExecutorDBTest::CompileExecutable(
      m_edb, m_input_shapes, m_input_graph.get(), m_ng_function, m_ng_exec,
      m_signature));
  std::shared_ptr<ngraph::runtime::Executable> evicted_ng_exec;
  std::shared_ptr<ngraph::Function> ng_function;
  m_edb.AddItem(m_signature, std::make_pair(evicted_ng_exec, ng_function),
                evicted_ng_exec, 2);
  ASSERT_EQ(m_edb.MaybeGetNgExecutable(m_signature, evicted_ng_exec), false);
  ASSERT_EQ(evicted_ng_exec.get(), nullptr);

  ASSERT_EQ(m_edb.MaybeGetNgExecutable(m_signature, m_ng_exec), false);
  ASSERT_EQ(m_edb.MaybeGetNgFunction(m_ng_exec, m_ng_function), false);

  m_edb.AddItem(m_signature, std::make_pair(m_ng_exec, m_ng_function),
                evicted_ng_exec, 2);
  ASSERT_EQ(m_edb.MaybeGetNgExecutable(m_signature, m_ng_exec), true);
  ASSERT_EQ(m_edb.MaybeGetNgFunction(m_ng_exec, m_ng_function), true);

  m_edb.AddItem(m_signature, std::make_pair(m_ng_exec, m_ng_function),
                evicted_ng_exec, 2);
  int a = m_edb.m_ng_exec_map.size();
  ASSERT_EQ(a, 1);
}

TEST_F(NGraphExecutorDBTest, CompileAndGetTensors) {
  ASSERT_EQ(m_edb.MaybeGetNgExecutable(m_signature, m_ng_exec), false);
  ASSERT_OK(NGraphExecutorDBTest::CompileExecutable(
      m_edb, m_input_shapes, m_input_graph.get(), m_ng_function, m_ng_exec,
      m_signature));
  std::shared_ptr<ngraph::runtime::Executable> evicted_ng_exec;
  m_edb.AddItem(m_signature, std::make_pair(m_ng_exec, m_ng_function),
                evicted_ng_exec, 2);
  ASSERT_EQ(m_edb.MaybeGetNgExecutable(m_signature, m_ng_exec), true);
  ASSERT_EQ(m_edb.MaybeGetNgFunction(m_ng_exec, m_ng_function), true);

  ASSERT_EQ(evicted_ng_exec.get(), nullptr);
  ng::runtime::Backend* op_backend = BackendManager::GetBackend("INTERPRETER");
  op_backend->remove_compiled_function(evicted_ng_exec);
  std::tuple<int, PipelinedTensorVector, PipelinedTensorVector> io_tensors;

  for (int i = 0; i < 2; i++) {
    ASSERT_OK(m_edb.GetDeviceTensors(m_ng_exec, io_tensors));
    int pipeline_idx = get<0>(io_tensors);
    ASSERT_EQ(i, pipeline_idx) << "GetTensorsFromPipeline() Returned: "
                               << pipeline_idx;
  }
}

TEST_F(NGraphExecutorDBTest, CompileAndGetTensorsMultiThreaded) {
  std::atomic_flag lock_stream = ATOMIC_FLAG_INIT;

  auto worker = [&](size_t thread_id) {
    ASSERT_OK(NGraphExecutorDBTest::CompileExecutable(
        m_edb, m_input_shapes, m_input_graph.get(), m_ng_function, m_ng_exec,
        m_signature));

    std::shared_ptr<ngraph::runtime::Executable> evicted_ng_exec;
    m_edb.AddItem(m_signature, std::make_pair(m_ng_exec, m_ng_function),
                  evicted_ng_exec, 4);
    int a = m_edb.m_ng_exec_map.size();
    ASSERT_EQ(a, 1);

    ASSERT_EQ(m_edb.MaybeGetNgExecutable(m_signature, m_ng_exec), true);
    ASSERT_EQ(m_edb.MaybeGetNgFunction(m_ng_exec, m_ng_function), true);

    ASSERT_EQ(evicted_ng_exec.get(), nullptr);
    ng::runtime::Backend* op_backend =
        BackendManager::GetBackend("INTERPRETER");
    op_backend->remove_compiled_function(evicted_ng_exec);
    std::tuple<int, PipelinedTensorVector, PipelinedTensorVector> io_tensors;
    for (int i = 0; i < 4; i++) {
      if (!lock_stream.test_and_set()) {
        ASSERT_OK(m_edb.GetDeviceTensors(m_ng_exec, io_tensors));
      }
    }
  };
  std::thread thread0(worker, 0);
  std::thread thread1(worker, 1);
  std::thread thread2(worker, 2);

  thread0.join();
  thread1.join();
  thread2.join();
}
}
}
}
