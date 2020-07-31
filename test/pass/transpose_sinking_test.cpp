//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <list>
#include <memory>

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/pass.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/util.hpp"

#include "logging/tf_graph_writer.h"
#include "ngraph_bridge/ngraph_utils.h"
#include "ngraph_bridge/pass/transpose_sinking.h"
#include "test/opexecuter.h"
#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

TEST(TransposeSinking, PassProperty) {
  auto pass = std::make_shared<TransposeSinking>();
  ASSERT_TRUE(
      pass->get_property(ngraph::pass::PassProperty::REQUIRE_STATIC_SHAPE));
  ASSERT_FALSE(
      pass->get_property(ngraph::pass::PassProperty::CHANGE_DYNAMIC_STATE));
}

TEST(TransposeSinking, EdgeSplitting) {
  // checks if Transpose is pushed through op::Abs, but stopped by ReduceSum
  ng::Shape shape_nhwc{16, 28, 28, 1};
  ng::Shape shape_nchw{16, 1, 28, 28};

  auto a = make_shared<ng::opset3::Parameter>(ng::element::i32, shape_nhwc);
  auto ng_order = std::make_shared<ngraph::opset3::Constant>(
      ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 3, 1, 2});
  auto transpose = make_shared<ng::opset3::Transpose>(a, ng_order);
  auto absn = make_shared<ng::opset3::Abs>(transpose);
  auto absn2 = make_shared<ng::opset3::Abs>(absn);

  auto axes = make_shared<ng::opset3::Constant>(ng::element::i64, ng::Shape{4},
                                                vector<int64_t>{0, 1, 2, 3});
  auto sum = make_shared<ng::opset3::ReduceSum>(transpose, axes, true);

  auto func = make_shared<ng::Function>(ng::OutputVector{absn2, sum},
                                        ng::ParameterVector{a});
  ng::pass::Manager pass_manager;
  pass_manager.register_pass<TransposeSinking>();
  pass_manager.run_passes(func);
  ASSERT_EQ(func->get_results().at(1)->get_argument(0), sum);
  auto new_transpose = ng::as_type_ptr<ng::opset3::Transpose>(
      func->get_results().at(0)->get_argument(0));
  ASSERT_TRUE(new_transpose);
  ASSERT_EQ(new_transpose->get_output_shape(0), shape_nchw);
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow