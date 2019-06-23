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
#ifndef _CONFIG_SETTING_H_
#define _CONFIG_SETTING_H_

#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace config_setting {

struct model {
  int id = 0;
  std::string graph;
  std::string backend;
  std::string input_tensor;
  std::string output_tensor;
  std::string image;
  std::string label_file;
  bool use_NCHW;
  int mean;
  int std;
  int height;
  int width;
  int chan;
  int batch_size;
};

struct profile {
  std::string name;
  std::vector<int> order;
  int loop;
};

class ConfigSetting {
 public:
  static ConfigSetting* getInstance() {
    instance = new ConfigSetting();
    return instance;
  }

  void ParseJsonFile(const std::string& file_name);
  int GetModelSize() { return m_models.size(); }
  int GetProfileSize() { return m_profiles.size(); }
  config_setting::model GetModel(int id) { return m_models[id]; }
  config_setting::profile GetProfile(int id) { return m_profiles[id]; }

 private:
  static ConfigSetting* instance;
  ConfigSetting(){};
  ~ConfigSetting() {
    m_models.clear();
    m_profiles.clear();
  }
  void PrintModels();
  void PrintProfiles();
  std::vector<config_setting::model> m_models;
  std::vector<config_setting::profile> m_profiles;
};

}  // namespace config_setting

#endif  // _CONFIG_SETTING_H_
