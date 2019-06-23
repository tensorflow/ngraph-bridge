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
#ifndef _INFERENCE_ENGINE_H_
#define _INFERENCE_ENGINE_H_

#include <stddef.h>
#include <unistd.h>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "config_setting.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/public/session.h"

using tensorflow::Status;
using tensorflow::Session;
using tensorflow::Tensor;

using std::string;
using std::unique_ptr;
using std::thread;
using std::atomic;
using std::function;

namespace infer_multiple_networks {

template <class T>
class InferQueue {
 public:
  InferQueue(void) : q(), m(), c(), m_stop(false) {}
  ~InferQueue(void) {}

  void push(const T& elem) {
    std::unique_lock<std::mutex> lock(m);
    q.push(elem);
    lock.unlock();
    c.notify_one();
  }

  bool pop(T& elem) {
    std::unique_lock<std::mutex> lock(m);
    while (!m_stop && q.empty()) {
      c.wait(lock);
    }
    if (!m_stop) {
      elem = q.front();
      q.pop();
      return true;
    }
    return false;
  }

  void stop() {
    m_stop = true;
    c.notify_all();
  }

 private:
  std::queue<T> q;
  mutable std::mutex m;
  std::condition_variable c;
  std::atomic<bool> m_stop;
};

/* This class contains session info, graph info
  one class per each session, each graph
  this could be shared between thread */
class InferSession {
 public:
  Tensor& GetNextImage();
  int GetId() { return m_id; }
  std::string GetName() { return m_model.graph; }
  InferSession();
  ~InferSession();
  Status LoadNetwork(const config_setting::model& model_config);
  Status LoadImages(const config_setting::model& model_config);
  Status Run();

 protected:
  static int sess_id;

 private:
  std::string GetInputLayer() { return m_model.input_tensor; }
  std::string GetOutputLayer() { return m_model.output_tensor; }
  std::string GetLabelFile() { return m_model.label_file; }
  int GetBatchSize() { return m_model.batch_size; }
  void ThreadOutput();
  int m_id;
  unique_ptr<Session> m_session;
  config_setting::model m_model;
  // std::queue<std::pair<int, Tensor>> inputs;
  InferQueue<std::pair<int, std::vector<Tensor>>> m_outputs_queue;
  thread m_output_worker;
  Tensor m_image_to_repeat;
  int m_expected_label_index;
  int m_img_num;
  atomic<bool> m_terminate{false};
};

class InferenceManager;

/* This class is for infer thread
  each Inference Engine will own one thread and can run multiple sessions
  */
class InferenceEngine {
 public:
  InferenceEngine(const config_setting::profile& profile);
  ~InferenceEngine();

  Status Start();
  Status Start(InferSession& sess);
  Status Start(const std::function<void(int)>& step_callback);
  Status Stop();

 private:
  InferenceManager* m_manager;
  InferSession* m_sess;
  const string m_name;
  config_setting::profile m_profile;
  void ThreadMain();
  thread m_worker;
  atomic<bool> m_terminate_worker{false};
  std::function<void(int)> m_step_callback{nullptr};
};

/* This is Inference Manager class which owns config/session/graph information
   This will invoke engine instance based on configuration
   */
class InferenceManager {
 public:
  static InferenceManager* getInstance() {
    if (instance == NULL) {
      instance = new InferenceManager();
    }
    return instance;
  }

  Status LoadConfig(const std::string& file_name);
  Status LoadNetworks();
  Status Start();
  Status WaitForDone();
  InferSession* GetSessionObj(int id) {
    if (m_graphs.size() <= id) std::cout << "graph id not exist" << std::endl;
    return m_graphs[id];
  }
  // Status Stop();

 private:
  InferenceManager(){};
  static InferenceManager* instance;

  config_setting::ConfigSetting* m_config;
  std::vector<InferSession*> m_graphs;
  // std::vector<config_setting::profile> m_models;
  // std::vector<config_setting::profile> m_profiles;
  std::vector<InferenceEngine*> m_engines;
};

}  // namespace infer_multiple_networks

#endif  // _INFERENCE_ENGINE_H_
