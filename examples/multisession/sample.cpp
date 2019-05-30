#include <memory>
#include <string>
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

#include "ngraph_backend_manager.h"

using namespace tensorflow;

int main(int argc, char** argv) {
  // Construct your graph.
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();

  // Matrix A = [3 2; -1 0]
  auto A = tensorflow::ops::Const(root, {{0.03f, 0.022f}, {-0.001f, 0.025f}});
  // Vector b = [3 5]
  auto b = tensorflow::ops::Const(root, {{0.345f, 0.35f}});
  // v = Ab^T
  auto v = tensorflow::ops::MatMul(root.WithOpName("v"), A, b,
                                   tensorflow::ops::MatMul::TransposeB(true));
  // R = softmax(v)
  auto R = tensorflow::ops::Softmax(root, v);

  // Turn off optimizations so that all the nodes are processed
  tensorflow::SessionOptions options;
  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(tensorflow::OptimizerOptions_Level_L0);
  options.config.mutable_graph_options()
      ->mutable_rewrite_options()
      ->set_constant_folding(tensorflow::RewriterConfig::OFF);

  options.config.mutable_graph_options()
      ->mutable_rewrite_options()
      ->add_custom_optimizers()
      ->set_name("ngraph-optimizer");

  options.config.mutable_graph_options()
    ->mutable_rewrite_options()->set_min_graph_nodes(-1);

  auto supported_backends =
      tensorflow::ngraph_bridge::BackendManager::GetSupportedBackendNames();
  vector<string> backends(supported_backends.begin(), supported_backends.end());

  cout << "Available backends: " << endl;
  for (auto& backend_name : backends) {
    cout << "Backend: " << backend_name << std::endl;
  }

  std::cout
      << "Currently selected backend: "
      << tensorflow::ngraph_bridge::BackendManager::GetCurrentlySetBackendName()
      << std::endl;

  tensorflow::ClientSession session(root, options);

  std::vector<tensorflow::Tensor> outputs;
  session.Run({R}, &outputs);

  // Print the output
  std::cout << "Result: " << outputs[0].matrix<float>() << std::endl;

  // Close the session.
  //session.Close();

  return 0;
}
