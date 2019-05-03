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

#include <mutex>
#include <thread>
#include <vector>

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

#include "ngraph/event_tracing.hpp"
#include "ngraph_backend_manager.h"
#include "ngraph_input_data_pipeline.h"

using namespace std;

void create_input_tensors() {
  // Create the inputs for this graph
}

void RunModel() {
  // Load the graph
  tensorflow::GraphDef gdef;
  std::cout << "Loading the input graph" << std::endl;

  if (tensorflow::ReadTextProto(tensorflow::Env::Default(), "test_axpy.pbtxt",
                                &gdef) != tensorflow::Status::OK()) {
    std::cout << "Failed to load the graph " << std::endl;
    return;
  }

  // create input TF Tensors
  tensorflow::Tensor x(tensorflow::DT_FLOAT, tensorflow::TensorShape({2, 3}));
  auto x_flat = x.flat<float>();
  for (int i = 0; i < x_flat.size(); i++) {
    x_flat.data()[i] = 1.0;
  }

  tensorflow::Tensor y(tensorflow::DT_FLOAT, tensorflow::TensorShape({2, 3}));
  auto y_flat = y.flat<float>();
  for (int i = 0; i < y_flat.size(); i++) {
    y_flat.data()[i] = 1.0;
  }

  // input node names
  vector<string> input_node_names = {"x", "y"};
  vector<tensorflow::Tensor*> input_tensors = {&x, &y};

  if (tensorflow::ngraph_bridge::NGraphInputDataPiepline::LoadInputDataOnDevice(
          input_node_names, input_tensors) != tensorflow::Status::OK()) {
    std::cout << "Failed to load the data on device " << std::endl;
    return;
  }

  std::vector<tensorflow::Tensor> outputs;

  tensorflow::SessionOptions options;
  tensorflow::ConfigProto& config = options.config;
  std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));

  session->Create(gdef);

  std::cout << "Created Session" << std::endl;
  session->Run({{"x", x}, {"y", y}}, {"mul", "add"}, {}, &outputs);

  for (int i = 0; i < outputs.size(); i++) {
    cout << "output " << i << endl;
    cout << outputs[i].SummarizeValue(64) << endl;
  }
  return;
}
