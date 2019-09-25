/*******************************************************************************
 * Copyright 2017-2019 Intel Corporation
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

#include "gtest/gtest.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"

#include "ngraph/ngraph.hpp"
#include "ngraph_bridge/version.h"

// Define useful macros used by others
#if !defined(ASSERT_OK)
#define ASSERT_OK(x) ASSERT_EQ((x), ::tensorflow::Status::OK())
#endif

#if !defined(ASSERT_NOT_OK)
#define ASSERT_NOT_OK(x) ASSERT_NE((x), ::tensorflow::Status::OK());
#endif

using namespace std;
namespace ng = ngraph;
namespace tf = tensorflow;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

class GenericUtil {
 public:
  static Status LoadGraphFromPbTxt(const string& pb_file,
                                   unique_ptr<tf::Graph>& new_graph) {
    // Read the graph
    tensorflow::GraphDef graph_def;
    auto load_graph_status = ReadTextProto(Env::Default(), pb_file, &graph_def);
    if (!load_graph_status.ok()) {
      return load_graph_status;
    }

    GraphConstructorOptions opts;
    opts.allow_internal_ops = true;
    unique_ptr<tf::Graph> input_graph =
        unique_ptr<tf::Graph>(new tf::Graph(OpRegistry::Global()));

    auto status = ConvertGraphDefToGraph(opts, graph_def, input_graph.get());
    new_graph = move(input_graph);
    return status;
  }
};
}  // namespace testing

}  // namespace ngraph_bridge

}  // namespace tensorflow
