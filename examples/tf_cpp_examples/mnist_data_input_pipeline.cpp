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
void AssignInputValuesRandom(tf::Tensor& A, T min, T max) {
  auto A_flat = A.flat<T>();
  auto A_flat_data = A_flat.data();
  srand(static_cast<unsigned>(time(0)));
  for (int i = 0; i < A_flat.size(); i++) {
    T value =
        // randomly generate a number between 0 and (max-min) inclusive
        static_cast<T>(rand()) / static_cast<T>(RAND_MAX / (max - min + 1));
    value = value + min;  // transform the range to (min, max) inclusive
    A_flat_data[i] = value;
  }
}

void LoadMNISTData() {
  for (int iteration = 0; iteration < 10; iteration++) {
    cout << "Loading data for iteration " << iteration << endl;

    vector<string> input_node_names = {"_arg_Placeholder_0_0"};

    tf::Tensor x(tf::DT_FLOAT, tf::TensorShape({16, 28, 28, 1}));
    AssignInputValuesRandom<float>(x, 0.0f, 255.0f);

    vector<tf::Tensor*> input_tensors = {&x};
    tf::ngraph_bridge::NGraphInputDataPiepline::LoadInputDataOnDevice(
        input_node_names, input_tensors);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

void ProcessMNISTData() {
  tf::GraphDef gdef;
  tf::ReadBinaryProto(
      tf::Env::Default(),
      "mnist_inference_quantized_trained_12212018.pb", &gdef);

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
    tf::Tensor x(tf::DT_FLOAT, tf::TensorShape({16, 28, 28, 1}));
    AssignInputValuesRandom<float>(x, 0.0f, 255.0f);

    ng_session->Run({{"Placeholder", x}},
                    {"softmax/quant/QuantizeAndDequantizeV2"}, {}, &ng_outputs);

    cout << "executed iteration " << iteration << endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

void RunMNISTInputDataPipelineExample() {
  cout << " Running MNIST input data pipeline example C++ " << endl;

  thread producer(LoadMNISTData);
  thread consumer(ProcessMNISTData);

  producer.join();
  consumer.join();

  cout << "exiting" << endl;
}