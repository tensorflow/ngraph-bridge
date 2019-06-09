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

#include <iostream>

#include "inference_engine.h"
#include "version.h"

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

using tensorflow::SessionOptions;
using tensorflow::RewriterConfig;
using tensorflow::OptimizerOptions_Level_L0;
using std::cout;

namespace tf = tensorflow;

extern tf::Status LoadGraph(const string& graph_file_name,
                            std::unique_ptr<tf::Session>* session,
                            const tf::SessionOptions& options);

namespace infer_multiple_networks {

InferenceEngine::InferenceEngine(const string& backend) {}
Status InferenceEngine::Load(const string& network, int input_width,
                             int input_height, float input_mean,
                             float input_std, const string& input_layer,
                             const string& output_layer, bool use_NCHW) {
  // Load the network
  TF_CHECK_OK(CreateSession(network, m_session));

  return Status::OK();
}

Status InferenceEngine::CreateSession(const string& graph_filename,
                                      unique_ptr<Session>& session) {
  SessionOptions options;
  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(OptimizerOptions_Level_L0);
  options.config.mutable_graph_options()
      ->mutable_rewrite_options()
      ->set_constant_folding(RewriterConfig::OFF);

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

    options.config.mutable_graph_options()
        ->mutable_rewrite_options()
        ->set_meta_optimizer_iterations(RewriterConfig::ONE);
  }

  // Load the network
  Status load_graph_status = LoadGraph(graph_filename, &session, options);
  return load_graph_status;
}

}  // namespace infer_multiple_networks
