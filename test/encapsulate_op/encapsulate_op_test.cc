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

#include "gtest/gtest.h"
#include "tensorflow/core/graph/node_builder.h"

#include "../test_utilities.h"
#include "ngraph_encapsulate_impl.h"
#include "ngraph_encapsulate_op.h"
#include "ngraph_utils.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {
#define ASSERT_OK(x) ASSERT_EQ((x), ::tensorflow::Status::OK());
#define ASSERT_NOT_OK(x) ASSERT_NE((x), ::tensorflow::Status::OK());

// Test: Allocating ngraph input tensors
TEST(EncapsulateOp, AllocateNGInputTensors) {
  NGraphEncapsulateImpl ng_encap_impl("AllocateNGInputTensors");
  ng::Shape shape{100};
  auto A = make_shared<ng::op::Parameter>(ng::element::f32, shape);
  auto B = make_shared<ng::op::Parameter>(ng::element::f32, shape);
  auto f = make_shared<ng::Function>(make_shared<ng::op::Add>(A, B),
                                     ng::ParameterVector{A, B});

  std::shared_ptr<ng::runtime::Backend> backend =
      ng::runtime::Backend::create("CPU");
  auto ng_exec = backend->compile(f);

  std::vector<tensorflow::TensorShape> input_shapes;
  std::vector<tensorflow::Tensor> input_tensors;
  input_shapes.push_back({0});
  input_shapes.push_back({2});
  input_shapes.push_back({6, 10});
  input_shapes.push_back({10, 10, 10});

  // create tensorflow tensors
  for (auto const& shapes : input_shapes) {
    Tensor input_data(DT_FLOAT, TensorShape(shapes));
    AssignInputValuesRandom<float>(input_data, -10.0, 20.0f);
    input_tensors.push_back(input_data);
  }

  std::vector<shared_ptr<ng::runtime::Tensor>> ng_inputs;

  ASSERT_OK(ng_encap_impl.AllocateNGInputTensors(input_tensors, ng_exec,
                                                 backend.get(), ng_inputs));
}

// Test: Allocating ngraph output tensors
TEST(EncapsulateOp, AllocateNGOutputTensors) {
  NGraphEncapsulateImpl ng_encap_impl("AllocateNGOutputTensors");
  ng::Shape shape{100};
  auto A = make_shared<ng::op::Parameter>(ng::element::f32, shape);
  auto B = make_shared<ng::op::Parameter>(ng::element::f32, shape);
  auto f = make_shared<ng::Function>(make_shared<ng::op::Add>(A, B),
                                     ng::ParameterVector{A, B});

  std::shared_ptr<ng::runtime::Backend> backend =
      ng::runtime::Backend::create("CPU");
  auto ng_exec = backend->compile(f);

  std::vector<tensorflow::TensorShape> input_shapes;
  std::vector<tensorflow::Tensor> outputs;
  input_shapes.push_back({0});
  input_shapes.push_back({2});
  input_shapes.push_back({6, 10});
  input_shapes.push_back({10, 10, 10});

  // Create tensorflow tensors
  for (auto const& shapes : input_shapes) {
    Tensor input_data(DT_FLOAT, TensorShape(shapes));
    AssignInputValuesRandom<float>(input_data, -10.0, 20.0f);
    outputs.push_back(input_data);
  }

  std::vector<tensorflow::Tensor*> output_tensors;
  for (int i = 0; i < outputs.size(); i++) {
    Tensor* output_tensor = &outputs[i];
    output_tensors.push_back(output_tensor);
  }
  std::vector<shared_ptr<ng::runtime::Tensor>> ng_outputs;

  ASSERT_OK(ng_encap_impl.AllocateNGOutputTensors(output_tensors, ng_exec,
                                                  backend.get(), ng_outputs));
}
}
}
}
