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

#pragma once

#include <fstream>
#include <ostream>
#include <sstream>

#include "ngraph/chrome_trace.hpp"
#include "ngraph/ngraph.hpp"

#include "log.h"

// Activates event logging until the end of the current code-block scoping;
// Automatically writes log data as soon as the the current scope expires.
#define NG_TRACE(name, category, args) \
  ngraph::event::Duration dx__ { (name), (category), (args) }

using namespace std;

namespace tensorflow {
namespace ngraph_bridge {
namespace utils {

// Collect the total memory usage through /proc/self/stat
void MemoryProfile(long&, long&);

// Check if we're supposed to dump graphs
bool DumpAllGraphs();

// Dump nGraph graphs in .dot format
void DumpNGGraph(std::shared_ptr<ngraph::Function> function,
                 const string filename_prefix);

// Get an environment variable
string GetEnv(const char* env);

// Set the environment variable env with val
void SetEnv(const char* env, const char* val);

}  // namespace utils
}  // namespace ngraph_bridge
}  // namespace tensorflow
