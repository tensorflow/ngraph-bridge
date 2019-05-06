/*******************************************************************************
 * Copyright 2017-2018 Intel Corporation
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

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/public/session.h"
#include "tf_graph_writer.h"

#include "ngraph_assign_clusters.h"
#include "ngraph_input_data_pipeline.h"
#include "ngraph_utils.h"
#include "test_utilities.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

#define ASSERT_OK(x) ASSERT_EQ((x), ::tensorflow::Status::OK());

TEST(DataLoading, AXPY) {
  // Load the graph
  GraphDef gdef;
  ASSERT_OK(ReadTextProto(Env::Default(), "test_axpy.pbtxt", &gdef));
  // create input TF Tensors
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

  // Session Config
  SessionOptions options;
  ConfigProto& config = options.config;

  // Run on nGraph, Load data on device first
  ActivateNGraph();

  // HARDCODED: input node names
  // In the graph test_axpy.pbtxt, the input nodes ("x" and "y")are placeholders
  // TF replaces them with _Arg nodes and renames them
  vector<string> input_node_names = {"_arg_x_0_0", "_arg_y_0_1"};
  vector<Tensor*> input_tensors = {&x, &y};
  ASSERT_OK(NGraphInputDataPiepline::LoadInputDataOnDevice(input_node_names,
                                                           input_tensors));

  std::unique_ptr<Session> ng_session(NewSession(options));
  ng_session->Create(gdef);
  std::vector<Tensor> ng_outputs;
  ng_session->Run({{"x", x}, {"y", y}}, {"mul", "add"}, {}, &ng_outputs);

  // Run on TF
  DeactivateNGraph();

  std::unique_ptr<Session> tf_session(NewSession(options));
  tf_session->Create(gdef);
  std::vector<Tensor> tf_outputs;
  tf_session->Run({{"x", x}, {"y", y}}, {"mul", "add"}, {}, &tf_outputs);

  // Compare the results
  Compare(tf_outputs, ng_outputs);
}

}  // namespace testing

}  // namespace ngraph_bridge

}  // namespace tensorflow
