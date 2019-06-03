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

#include "ngraph_backend_manager.h"
#include "version.h"

using namespace std;
namespace tf = tensorflow;

extern tf::Status LoadGraph(const string& graph_file_name,
                            std::unique_ptr<tf::Session>* session,
                            const tf::SessionOptions& options);

extern tf::Status ReadTensorFromImageFile(const string& file_name,
                                          const int input_height,
                                          const int input_width,
                                          const float input_mean,
                                          const float input_std,
                                          std::vector<tf::Tensor>* out_tensors);

extern tf::Status PrintTopLabels(const std::vector<tf::Tensor>& outputs,
                      const string& labels_file_name);


// Prints the available backends
void PrintAvailableBackends() {
  // Get the list of backends
  auto supported_backends =
      tf::ngraph_bridge::BackendManager::GetSupportedBackendNames();
  vector<string> backends(supported_backends.begin(), supported_backends.end());

  cout << "Available backends: " << endl;
  for (auto& backend_name : backends) {
    cout << "Backend: " << backend_name << std::endl;
  }
}

// Sets the specified backend. This backend must be set BEFORE running
// the computation
tf::Status SetNGraphBackend(const string& backend_name) {
  // Select a backend
  tf::Status status =
      tf::ngraph_bridge::BackendManager::SetBackendName(backend_name);
  return status;
}

// Create a simple computation graph and run
void RunSimpleNetworkExample() {
  // Create the graph
  tf::Scope root = tf::Scope::NewRootScope();

  // Matrix A = [3 2; -1 0]
  auto A = tf::ops::Const(root, {{0.03f, 0.022f}, {-0.001f, 0.025f}});
  // Vector b = [3 5]
  auto b = tf::ops::Const(root, {{0.345f, 0.35f}});
  // v = Ab^T
  auto v = tf::ops::MatMul(root.WithOpName("v"), A, b,
                           tf::ops::MatMul::TransposeB(true));
  // R = softmax(v)
  auto R = tf::ops::Softmax(root, v);

  // Turn off optimizations so that all the nodes are processed
  tf::SessionOptions options;
  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(tf::OptimizerOptions_Level_L0);
  options.config.mutable_graph_options()
      ->mutable_rewrite_options()
      ->set_constant_folding(tf::RewriterConfig::OFF);

  // The following is related to Grapller - which we are turning off
  // Until we get a library fully running
  if (tf::ngraph_bridge::ngraph_tf_is_grappler_enabled()) {
    options.config.mutable_graph_options()
        ->mutable_rewrite_options()
        ->add_custom_optimizers()
        ->set_name("ngraph-optimizer");

    options.config.mutable_graph_options()
        ->mutable_rewrite_options()
        ->set_min_graph_nodes(-1);
  }

  std::cout << "Currently selected backend: "
            << tf::ngraph_bridge::BackendManager::GetCurrentlySetBackendName()
            << std::endl;

  tf::ClientSession session(root, options);

  std::vector<tf::Tensor> outputs;
  session.Run({R}, &outputs);

  // Print the output
  std::cout << "Result: " << outputs[0].matrix<float>() << std::endl;
}

void PrintVersion() {
  // nGraph Bridge version info
  std::cout << "Bridge version: " << tf::ngraph_bridge::ngraph_tf_version()
            << std::endl;
  std::cout << "nGraph version: " << tf::ngraph_bridge::ngraph_lib_version()
            << std::endl;
  std::cout << "CXX11_ABI Used: "
            << tf::ngraph_bridge::ngraph_tf_cxx11_abi_flag() << std::endl;
  std::cout << "Grappler Enabled? "
            << (tf::ngraph_bridge::ngraph_tf_is_grappler_enabled()
                    ? std::string("Yes")
                    : std::string("No"))
            << std::endl;
  std::cout << "Variables Enabled? "
            << (tf::ngraph_bridge::ngraph_tf_are_variables_enabled()
                    ? std::string("Yes")
                    : std::string("No"))
            << std::endl;

  PrintAvailableBackends();
}

std::unique_ptr<tf::Session> CreateSession(const string& filename) {
  tf::SessionOptions options;
  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(tf::OptimizerOptions_Level_L0);
  options.config.mutable_graph_options()
      ->mutable_rewrite_options()
      ->set_constant_folding(tf::RewriterConfig::OFF);

  // The following is related to Grapller - which we are turning off
  // Until we get a library fully running
  if (tf::ngraph_bridge::ngraph_tf_is_grappler_enabled()) {
    options.config.mutable_graph_options()
        ->mutable_rewrite_options()
        ->add_custom_optimizers()
        ->set_name("ngraph-optimizer");

    options.config.mutable_graph_options()
        ->mutable_rewrite_options()
        ->set_min_graph_nodes(-1);
  }

  // Load the network
  std::unique_ptr<tf::Session> session;
  tf::Status load_graph_status = LoadGraph(filename, &session, options);

  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return nullptr;
  }
  return std::move(session);
}

int main(int argc, char** argv) {
  const char* backend = "CPU";

  if (argc > 1) {
    backend = argv[1];
  }

  if (SetNGraphBackend(backend) != tf::Status::OK()) {
    std::cout << "Error: Cannot set the backend: " << backend << std::endl;
    return -1;
  }

  // Run the MatMul example
  auto session = CreateSession("inception_v3_2016_08_28_frozen.pb");

  // Get the image from disk as a float array of numbers, resized and normalized
  // to the specifications the main graph expects.
  std::vector<tf::Tensor> resized_tensors;
  tf::Status read_tensor_status = ReadTensorFromImageFile(
      "grace_hopper.jpg", 299 /*input_height*/, 299 /*input_width*/,
      0.0 /*input_mean*/, 255 /*input_std*/, &resized_tensors);
  if (!read_tensor_status.ok()) {
    LOG(ERROR) << read_tensor_status;
    return -1;
  }

  const tf::Tensor& resized_tensor = resized_tensors[0];
  string input_layer = "input";
  std::vector<tf::Tensor> outputs;
  string output_layer = "InceptionV3/Predictions/Reshape_1";
  tf::Status run_status = session->Run({{input_layer, resized_tensor}},
                                       {output_layer}, {}, &outputs);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  }

  // Do something interesting with the results we've generated.
  string labels = "imagenet_slim_labels.txt";
  tf::Status print_status = PrintTopLabels(outputs, labels);
  if (!print_status.ok()) {
    LOG(ERROR) << "Running print failed: " << print_status;
    return -1;
  }

  return 0;
}
