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

#include <chrono>
#include <iostream>
#include <sstream>
#include <vector>

#include "inference_engine.h"
#include "ngraph/event_tracing.hpp"
#include "ngraph/util.hpp"
#include "ngraph_backend_manager.h"
#include "version.h"

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

using tensorflow::SessionOptions;
using tensorflow::RewriterConfig;
using tensorflow::OptimizerOptions_Level_L0;
using tensorflow::Tensor;
using std::cout;
using std::move;
using std::ostringstream;
using namespace infer_multiple_networks;

namespace tf = tensorflow;

extern tf::Status LoadGraph(const string& graph_file_name,
                            std::unique_ptr<tf::Session>* session,
                            const tf::SessionOptions& options);

extern tf::Status ReadTensorFromImageFile(
    const string& file_name, const int input_chan, const int input_height,
    const int input_width, const float input_mean, const float input_std,
    bool use_NCHW, std::vector<tf::Tensor>* out_tensors);

extern tf::Status ReadTensorFromImageList(
    const std::vector<string>& file_name, const int input_chan,
    const int input_height, const int input_width, const float input_mean,
    const float input_std, bool use_NCHW, std::vector<Tensor>* out_tensors);

extern tf::Status PrintTopLabels(const std::vector<Tensor>& outputs,
                                 const string& labels_file_name,
                                 const int batch_size, const int max_count);

extern tf::Status CheckTopLabel(const std::vector<Tensor>& outputs,
                                int batch_size, int expected,
                                bool* is_expected);

namespace infer_multiple_networks {
InferenceManager* InferenceManager::instance = NULL;

//////////////////////////////
///  Class : InferSession
//////////////////////////////
int InferSession::sess_id = 0;

InferSession::InferSession() : m_expected_label_index(-1), m_img_num(0) {
  m_id = sess_id++;
  std::cout << "Session created [" << m_id << "]\n";

  // Create output label processing thread
  thread output_worker(&InferSession::ThreadOutput, this);
  m_output_worker = move(output_worker);
}
InferSession::~InferSession() {
  m_terminate = true;
  if (m_output_worker.joinable()) {
    m_outputs_queue.stop();
    m_output_worker.join();
  }
  std::cout << "Session deleted [" << m_id << "] \n";
}

Status InferSession::LoadNetwork(const config_setting::model& model) {
  // Create Session
  unique_ptr<Session> session;

  SessionOptions options;
  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(OptimizerOptions_Level_L0);
  options.config.mutable_graph_options()
      ->mutable_rewrite_options()
      ->set_constant_folding(RewriterConfig::OFF);

  // The following is related to Grappler - which we are turning off
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

  // Set backend
  tf::ngraph_bridge::BackendManager::SetBackendName(model.backend);

  // Load the network
  Status load_graph_status = LoadGraph(model.graph, &session, options);
  m_session = std::move(session);

  // Save model information
  m_model = model;

  return load_graph_status;
}

Status InferSession::LoadImages(const config_setting::model& model) {
  // Load Images
  // Set the CPU as the backend before these ops
  string current_backend =
      tf::ngraph_bridge::BackendManager::GetCurrentlySetBackendName();
  tf::ngraph_bridge::BackendManager::SetBackendName("CPU");
  std::vector<tf::Tensor> resized_tensors;

  // TODO: This can be moved to run in a seperate thread.
  cout << "Reading Image :" << model.image << "\n";
  ngraph::Event read_event("Read", "", "");

  string model_ext = model.image.substr(model.image.find_last_of(".") + 1);
  string image_name;
  if (model_ext != "txt") {
    image_name = model.image;
  } else {
    // TODO Read image file name from the list to image pool
    // Currently, just read the top image and reuse it.
    std::ifstream file(model.image);
    if (!file) {
      return tensorflow::errors::NotFound("Image list file ", model.image,
                                          " not found.");
    }
    std::string index, label, image_file;
    if (std::getline(file, index, ',')) cout << "index: " << index;
    if (std::getline(file, label, ',')) cout << ", label: " << label;
    if (std::getline(file, image_file))
      cout << ", image: " << image_file << endl;

    image_name = image_file;
    m_expected_label_index = std::stoi(index);
  }

  if (model.batch_size == 1) {
    TF_CHECK_OK(ReadTensorFromImageFile(image_name, model.chan, model.height,
                                        model.width, model.mean, model.std,
                                        model.use_NCHW, &resized_tensors));
  } else {
    std::vector<string> image_list;
    for (int i = 0; i < model.batch_size; i++) {
      image_list.push_back(image_name);
    }
    TF_CHECK_OK(ReadTensorFromImageList(image_list, model.chan, model.height,
                                        model.width, model.mean, model.std,
                                        model.use_NCHW, &resized_tensors));
  }

  m_image_to_repeat = resized_tensors[0];
  read_event.Stop();
  ngraph::Event::write_trace(read_event);

  tf::ngraph_bridge::BackendManager::SetBackendName(current_backend);

  return Status::OK();
}

Tensor& InferSession::GetNextImage() {
  // TODO get image pointer from image pool
  m_img_num++;
  return m_image_to_repeat;
}

void InferSession::ThreadOutput() {
  while (true) {
    std::pair<int, std::vector<Tensor>> outputs;
    bool result = m_outputs_queue.pop(outputs);
    if (result) {
      if (m_expected_label_index != -1) {
        bool is_correct;
        CheckTopLabel(outputs.second, GetBatchSize(), m_expected_label_index,
                      &is_correct);
        if (is_correct == false) {
          throw runtime_error("Label doesn't match!!!");
        }
      } else {
        // cout << "[sess-" << std::to_string(m_id) << "] " << outputs.first <<
        // ":" << endl;
        // PrintTopLabels(outputs.second, GetLabelFile(), GetBatchSize(), 1);
      }
    }

    if (m_terminate) break;
  }
}

Status InferSession::Run() {
  const tf::Tensor& resized_tensor = GetNextImage();
  std::vector<Tensor> outputs;
  ngraph::Event infer_event("Infer", "", "");
  TF_CHECK_OK(m_session->Run({{GetInputLayer(), resized_tensor}},
                             {GetOutputLayer()}, {}, &outputs));
  infer_event.Stop();
  ngraph::Event::write_trace(infer_event);
  m_outputs_queue.push(std::make_pair(m_img_num - 1, outputs));
  return Status::OK();
}

/////////////////////////////
///  Class : InferenceEngine
//////////////////////////////
InferenceEngine::InferenceEngine(const config_setting::profile& profile)
    : m_name(profile.name), m_profile(profile) {
  m_manager = InferenceManager::getInstance();
}

InferenceEngine::~InferenceEngine() {
  if (m_worker.joinable()) {
    m_worker.join();
  }
}

Status InferenceEngine::Start(const function<void(int)>& step_callback) {
  m_step_callback = step_callback;
  return Start();
}

Status InferenceEngine::Start() {
  thread new_worker(&InferenceEngine::ThreadMain, this);
  m_worker = move(new_worker);
  return Status::OK();
}

void InferenceEngine::ThreadMain() {
  std::vector<int> infer_model_order;

  for (int i = 0; i < m_profile.loop; i++) {
    infer_model_order.insert(std::end(infer_model_order),
                             std::begin(m_profile.order),
                             std::end(m_profile.order));
  }

  int step_count = 0;
  ngraph::stopwatch timer;
  timer.start();
  while (step_count < infer_model_order.size()) {
    ostringstream ss;
    ss << "[" << m_name << "] Iteration: " << step_count;
    ngraph::Event itreation_event(ss.str(), "", "");

    int model_num = infer_model_order.at(step_count);
    InferSession* sess = m_manager->GetSessionObj(model_num);
    cout << "[" << m_name << ", sess-" << sess->GetId() << "] " << step_count
         << ": Submit image for inference\n";
    sess->Run();

    step_count++;

    itreation_event.Stop();
    ngraph::Event::write_trace(itreation_event);
  }
  timer.stop();
  cout << "[" << m_name << "] done - " << setw(7) << timer.get_milliseconds()
       << "ms" << endl;
}

Status InferenceEngine::Stop() {
  cout << "[" << m_name << "] Stop called" << std::endl;

  if (m_worker.joinable()) {
    cout << "Join..." << endl;
    m_worker.join();
  }

  return Status::OK();
}

/////////////////////////////
///  Class : InferenceManager
//////////////////////////////
Status InferenceManager::LoadConfig(const string& filename) {
  // Parse Json file
  m_config = config_setting::ConfigSetting::getInstance();
  m_config->ParseJsonFile(filename);

  return Status::OK();
}

Status InferenceManager::LoadNetworks() {
  Status status = Status::OK();
  // Load all graphs
  // TODO: This could be also run in multiple thread for performance.
  // For now, we are loading all graphs sequentially prior to starting
  // inference.
  for (int i = 0; i < m_config->GetModelSize(); i++) {
    auto model = m_config->GetModel(i);
    cout << "Load Network : " << model.graph << endl;
    InferSession* sess = new InferSession();
    status = sess->LoadNetwork(model);
    status = sess->LoadImages(model);

    m_graphs.push_back(sess);

    // Do coldrun to compile/load the graph
    // TODO: Move this into a seperate thread for performance.
    cout << "[sess-" << sess->GetId() << "] Coldrun " << endl;
    sess->Run();
  }

  return status;
}

Status InferenceManager::Start() {
  for (int i = 0; i < m_config->GetProfileSize(); i++) {
    auto profile = m_config->GetProfile(i);
    infer_multiple_networks::InferenceEngine* engine =
        new infer_multiple_networks::InferenceEngine(profile);

    engine->Start();

    m_engines.push_back(engine);
  }

  return Status::OK();
}

Status InferenceManager::WaitForDone() {
  for (auto engine : m_engines) {
    delete engine;
  }
  for (auto sess : m_graphs) {
    delete sess;
  }

  m_engines.clear();
  m_graphs.clear();

  return Status::OK();
}

}  // namespace infer_multiple_networks
