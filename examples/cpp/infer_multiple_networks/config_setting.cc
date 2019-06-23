#include "config_setting.h"
#include <iostream>
#include "nlohmann/json.hpp"

using namespace std;
using namespace config_setting;
using json = nlohmann::json;

ConfigSetting* ConfigSetting::instance = NULL;

template <typename T>
T get_value(nlohmann::json js, const string& key) {
  T rc{};
  try {
    auto it = js.find(key);
    if (it != js.end()) {
      rc = it->get<T>();
    }
  } catch (...) {
    throw runtime_error("Error parsing json at node: " + key);
  }
  return rc;
}

void ConfigSetting::ParseJsonFile(const std::string& file_name) {
  std::cout << "ConfigSetting::ParseJsonFile" << endl;
  std::ifstream ifs(file_name);
  json node_js = json::parse(ifs);

  try {
    vector<json> models = get_value<vector<json>>(node_js, "models");
    vector<json> profiles = get_value<vector<json>>(node_js, "profiles");

    for (auto& model : models) {
      config_setting::model info;
      info.id = get_value<int>(model, "id");
      info.graph = get_value<string>(model, "graph");
      info.image = get_value<string>(model, "image_list");
      info.label_file = get_value<string>(model, "label_file");
      info.std = get_value<int>(model, "image_std");
      info.mean = get_value<int>(model, "image_mean");
      info.input_tensor = get_value<string>(model, "input_tensor_name");
      info.output_tensor = get_value<string>(model, "output_tensor_name");
      info.use_NCHW =
          get_value<string>(model, "image_format") == "NCHW" ? true : false;
      info.batch_size = get_value<int>(model, "batch_size");
      info.backend = get_value<string>(model, "backend");
      std::vector<int> dimensions =
          get_value<vector<int>>(model, "input_dimension");
      info.chan = dimensions[0];
      info.height = dimensions[1];
      info.width = dimensions[2];

      m_models.push_back(info);
    }

    for (auto& profile : profiles) {
      config_setting::profile info;
      info.name = get_value<string>(profile, "name");
      info.order = get_value<vector<int>>(profile, "model_order");
      info.loop = get_value<int>(profile, "loop");
      m_profiles.push_back(info);
    }

    cout << "Total models: " << m_models.size()
         << " , Total profile: " << m_profiles.size() << endl;
    PrintModels();
    PrintProfiles();

  } catch (...) {
    throw runtime_error("Error parsing json file " + file_name);
  }
}

void ConfigSetting::PrintModels() {
  for (auto model : m_models) {
    cout << "Model Info::"
         << "\n id: " << model.id << "\n graph: " << model.graph
         << "\n image: " << model.image << "\n label: " << model.label_file
         << "\n chan: " << model.chan << "\n height: " << model.height
         << "\n width: " << model.width << "\n use_NCHW: " << model.use_NCHW
         << "\n std: " << model.std << "\n mean: " << model.mean
         << "\n batch_size: " << model.batch_size
         << "\n backend: " << model.backend
         << "\n input_tensor: " << model.input_tensor << "\n output_tensor "
         << model.output_tensor;
  }
}

void ConfigSetting::PrintProfiles() {
  for (auto profile : m_profiles) {
    cout << "Profile Info::"
         << "\n name: " << profile.name << "\n order: [";
    for (int i = 0; i < profile.order.size(); i++) {
      cout << profile.order.at(i);
    }
    cout << "]\n loop: " << profile.loop << endl;
  }
}
