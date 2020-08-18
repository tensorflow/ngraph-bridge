/*******************************************************************************
 * Copyright 2017-2020 Intel Corporation
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

#include <dlfcn.h>
#include <stdlib.h>
#include <cctype>
#include <chrono>
#include <fstream>
#include <iostream>
#include <regex>

#include "gtest/gtest.h"

#include "tensorflow/core/platform/env.h"

#ifdef __APPLE__
#define EXT "dylib"
#else
#define EXT "so"
#endif

using namespace std;

static void set_filters_for_backend() {
  std::string backend = "GENERIC";
  char* pBE = getenv("NGRAPH_TF_BACKEND");
  if (pBE) {
    backend = std::string(pBE);
    // backend.replace(backend.begin(), backend.end(), ":", "");
    backend = std::regex_replace(backend, std::regex(":"), "");
  }
  std::string this_filepath = __FILE__;
  this_filepath =
      std::regex_replace(this_filepath, std::regex("^(.*)/.*$"), "$1");
  std::string filter_file =
      this_filepath + "/" + "cpp_tests_list_" + backend + ".txt";
  fstream fs;
  fs.open(filter_file, ios::in);
  if (fs.is_open()) {
    std::string filters = "";
    std::string line;
    bool excluded_section = false;
    while (getline(fs, line)) {
      if (line.empty()) continue;
      if (std::all_of(line.begin(), line.end(),
                      [](unsigned char c) { return std::isspace(c); }))
        continue;
      if (line.find("#") == 0) continue;
      if (!excluded_section && line.find("[EXCLUDED]") == 0) {
        filters += "-";
        excluded_section = true;
        continue;
      }
      if (excluded_section) {
        if (line.at(0) == '-') line = line.erase(0, 1);
      }
      filters += line + ":";
    }
    fs.close();
    if (filters.back() == ':') filters.pop_back();  // remove last :

    ::testing::GTEST_FLAG(filter) = filters;
  }
}

int main(int argc, char** argv) {
  set_filters_for_backend();

  ::testing::InitGoogleTest(&argc, argv);

  int rc = RUN_ALL_TESTS();
  return rc;
}
