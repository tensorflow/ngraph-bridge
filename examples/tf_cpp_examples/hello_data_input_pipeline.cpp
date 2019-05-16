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
#include <chrono>
#include <thread>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/platform/env.h"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

#include "ngraph_input_data_pipeline.h"

using namespace std;

namespace tf = tensorflow;

template <typename T>
void AssignInputValues(tf::Tensor& A, T x) {
  auto A_flat = A.flat<T>();
  auto A_flat_data = A_flat.data();
  for (int i = 0; i < A_flat.size(); i++) {
    A_flat_data[i] = x;
  }
}

void LoadData() {
  for (int iteration = 0; iteration < 10; iteration++) {
    cout << "Loading data for iteration " << iteration << endl;

    vector<string> input_node_names = {"_arg_x_0_0", "_arg_y_0_1"};

    tf::Tensor x(tf::DT_FLOAT, tf::TensorShape({2, 3}));
    AssignInputValues<float>(x, 1.0f);

    tf::Tensor y(tf::DT_FLOAT, tf::TensorShape({2, 3}));
    AssignInputValues<float>(y, 2.0f);

    vector<tf::Tensor*> input_tensors = {&x, &y};
    tf::ngraph_bridge::NGraphInputDataPiepline::LoadInputDataOnDevice(
        input_node_names, input_tensors);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

void ProcessData() {
  tf::GraphDef gdef;
  tf::ReadTextProto(tf::Env::Default(), "test_axpy.pbtxt", &gdef);

  // Create Session
  tf::SessionOptions options;
  tf::ConfigProto& config = options.config;
  std::unique_ptr<tf::Session> ng_session(NewSession(options));
  ng_session->Create(gdef);

  for (int iteration = 0; iteration < 10; iteration++) {
    cout << "begin executing iteration " << iteration << endl;
    std::vector<tf::Tensor> ng_outputs;
    // dummy data
    // These tensors are required for session.run call
    // but will not be actually used
    tf::Tensor x(tf::DT_FLOAT, tf::TensorShape({2, 3}));
    AssignInputValues<float>(x, 1.0f);

    tf::Tensor y(tf::DT_FLOAT, tf::TensorShape({2, 3}));
    AssignInputValues<float>(x, 1.0f);

    ng_session->Run({{"x", x}, {"y", y}}, {"mul", "add"}, {}, &ng_outputs);

    cout << "executed iteration " << iteration << endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

void RunInputDataPipelineExample() {
  cout << " Running input data pipeline example C++ " << endl;

  thread producer(LoadData);
  thread consumer(ProcessData);

  producer.join();
  consumer.join();

  cout << "exiting" << endl;
}