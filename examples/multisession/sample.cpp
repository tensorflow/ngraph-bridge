#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

using namespace tensorflow;

int main(int argc, char** argv) {
  // Construct your graph.
  tensorflow::GraphDef graph;  // = ...;

  // Create a Session running TensorFlow locally in process.
  std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession({}));

  // Initialize the session with the graph.
  tensorflow::Status s = session->Create(graph);
  // if (!s.ok()) { ... }

  // Specify the 'feeds' of your network if needed.
  std::vector<std::pair<string, tensorflow::Tensor>> inputs;

  // Run the session, asking for the first output of "my_output".
  std::vector<tensorflow::Tensor> outputs;
  s = session->Run(inputs, {"my_output:0"}, {}, &outputs);
// f (!s.ok()) { ... }

// Do something with your outputs
#if 0
  auto output_vector = outputs[0].vec<float>();
  if (output_vector(0) > 0.5) { ... }
#endif

  // Close the session.
  session->Close();

  return 0;
}
