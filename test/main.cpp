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
#include <array>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>

#include "gtest/gtest.h"

#include "tensorflow/core/platform/env.h"

#ifdef __APPLE__
#define EXT "dylib"
#else
#define EXT "so"
#endif

using namespace std;

string SCRIPTDIR =
    std::regex_replace(__FILE__, std::regex("^(.*)/[^/]+$"), "$1");

// will only get stdout back, not stderr
std::string exec_cmd(const char* cmd) {
  std::string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }
  std::array<char, 128> buffer;  // transfer at a time
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }
  return result;
}

// manifestfile must be in abs path format
static void read_tests_from_manifest(string manifestfile, set<string>& run_list,
                                     set<string>& skip_list) {
  static set<string> g_imported_files;
  fstream fs;
  fs.open(manifestfile, ios::in);
  if (fs.is_open()) {
    cout << "Parsing manifest: " << manifestfile << " ...\n";
    string line;
    string curr_section = "";
    g_imported_files.insert(manifestfile);
    while (getline(fs, line)) {
      line = std::regex_replace(line, std::regex("^\\s+"), "");
      line = std::regex_replace(line, std::regex("#.*$"), "");
      line = std::regex_replace(line, std::regex("\\s+$"), "");
      if (line.empty()) continue;
      if (std::regex_search(line, std::regex("^\\[IMPORT\\]$"))) {
        curr_section = "import_section";
        continue;
      }
      if (std::regex_search(line, std::regex("^\\[RUN\\]$"))) {
        curr_section = "run_section";
        continue;
      }
      if (std::regex_search(line, std::regex("^\\[SKIP\\]$"))) {
        curr_section = "skip_section";
        continue;
      }
      if (curr_section == "import_section") {
        if (g_imported_files.find(line) != g_imported_files.end()) {
          cout << "ERROR: re-import of manifest " << line << " in "
               << manifestfile << endl;
          exit(1);
        }
        line = SCRIPTDIR + "/" + line;
        g_imported_files.insert(line);
        set<string> new_runs, new_skips;
        read_tests_from_manifest(line, new_runs, new_skips);
        run_list.insert(new_runs.begin(), new_runs.end());
        skip_list.insert(new_skips.begin(), new_skips.end());
      }
      if (std::regex_search(line, std::regex("[:\\s]"))) {
        cout << "Bad pattern: [" << line << "], ignoring...\n";
        continue;
      }
      if (curr_section == "run_section") {
        run_list.insert(line);
      }
      if (curr_section == "skip_section") {
        skip_list.insert(line);
      }
    }
    fs.close();
  } else {
    cout << "Cannot open file: <" << manifestfile << ">\n";
  }
}

// To keep the logic in one place, we will invoke function from test_utils.py
static string get_test_manifest_filepath() {
  string cmdstr =
      "python3 " + SCRIPTDIR + "/../tools/test_utils.py --run_func print_test_manifest_filename";
  auto resp = exec_cmd(cmdstr.c_str());
  if (resp.back() == '\n') resp.pop_back();
  resp = std::regex_replace(resp, std::regex("^\\s+"), "");
  resp = std::regex_replace(resp, std::regex("\\s+$"), "");
  if (resp.front() != '/') {  // works in Linux
    resp = SCRIPTDIR + "/" + resp;
  }
  return resp;
}

static void set_filters_from_file() {
  string filter_file = get_test_manifest_filepath();
  set<string> run_list, skip_list;
  read_tests_from_manifest(filter_file, run_list, skip_list);

  string filters = "";
  for (auto& it : run_list) {
    filters += it + ":";
  }
  if (filters.back() == ':') filters.pop_back();  // remove last :
  if (skip_list.size() > 0)
    filters += "-";  // separator before the skips/excludes
  for (auto& it : skip_list) {
    filters += it + ":";
  }
  if (filters.back() == ':') filters.pop_back();  // remove last :

  ::testing::GTEST_FLAG(filter) = filters;
}

int main(int argc, char** argv) {
  bool filter_arg = false;
  for (int i = 1; i < argc; i++) {
    if (strncmp(argv[i], "--gtest_filter", 14) == 0) {
      filter_arg = true;
      break;
    }
  }
  // if(::testing::GTEST_FLAG(filter) == "*") {
  if (!filter_arg) {
    cout << "Will use test manifest to set test filters...\n";
    set_filters_from_file();
  }

  ::testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  return rc;
}
