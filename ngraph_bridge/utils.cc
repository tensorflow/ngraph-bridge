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

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "utils.h"
#include "version.h"

using namespace std;

namespace tensorflow {
namespace ngraph_bridge {
namespace utils {

void MemoryProfile(long& vm_usage, long& resident_set) {
  vm_usage = 0;
  resident_set = 0;

  // Get the two fields we want
  long vsize;
  long rss;

  std::ifstream ifs("/proc/self/stat", std::ios_base::in);
  std::string mem_in;
  getline(ifs, mem_in);
  if (mem_in != "") {
    vector<string> mem_str = ngraph::split(mem_in, ' ');
    vsize = std::stol(mem_str[22]);
    rss = std::stol(mem_str[23]);

    long page_size_kb = sysconf(_SC_PAGE_SIZE) /
                        1024;  // in case x86-64 is configured to use 2MB pages
    vm_usage = vsize / 1024;   // unit kb
    resident_set = rss * page_size_kb;
  }
}

void DumpNGGraph(std::shared_ptr<ngraph::Function> function,
                 const string filename) {
  if (!DumpAllGraphs()) {
    return;
  }

  NGRAPH_VLOG(0) << "Dumping nGraph graph to " << filename + ".dot";
  // enable shape info for nGraph graphs
  SetEnv("NGRAPH_VISUALIZE_TREE_OUTPUT_SHAPES", "1");
  SetEnv("NGRAPH_VISUALIZE_TREE_OUTPUT_TYPES", "1");
  SetEnv("NGRAPH_VISUALIZE_TREE_IO", "1");
  ngraph::plot_graph(function, filename + ".dot");
}

bool DumpAllGraphs() { return GetEnv("NGRAPH_TF_DUMP_GRAPHS") == "1"; }

string GetEnv(const char* env) {
  const char* val = std::getenv(env);
  if (val == nullptr) {
    return "";
  } else {
    return string(val);
  }
}

void SetEnv(const char* env, const char* val) { setenv(env, val, 1); }

}  // namespace utils
}  // namespace ngraph_bridge
}  // namespace tensorflow
