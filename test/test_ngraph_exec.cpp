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

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/platform/env.h"

#include "ngraph_bridge/ngraph_builder.h"
#include "ngraph_bridge/ngraph_pipelined_tensors.h"
#include "ngraph_bridge/ngraph_utils.h"

#include "test_utilities.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

class NGraphExecTest : public ::testing::Test {
 protected:
  Status LoadGraph(string graph_pbtxt_file, Graph* graph) {
    GraphDef gdef;
    TF_RETURN_IF_ERROR(ReadTextProto(Env::Default(), graph_pbtxt_file, &gdef));
    GraphConstructorOptions opts;
    // Set the allow_internal_ops to true so that graphs with node names such as
    // _arg_Placeholder_1_0_1_0_arg are allowed. These op names are generated
    // during the graph rewrite passes and considered internal
    TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(opts, gdef, graph));
    return Status::OK();
  }

  void OverrideBackendFromEnv(string* backend_name) {
    string env_name = "NGRAPH_TF_BACKEND";
    if (IsEnvVariableSet(env_name)) {
      *backend_name = GetEnvVariable(env_name);
    }
  }

  tuple<PipelinedTensorMatrix, PipelinedTensorMatrix> CreatePipelinedTensors(
      const shared_ptr<ngraph::runtime::Executable> ng_exec,
      const vector<int> pipelined_input_indexes,
      const vector<int> pipelined_output_indexes, const int pipeline_depth) {
    PipelinedTensorMatrix inputs(pipelined_input_indexes.size());
    PipelinedTensorMatrix outputs(pipelined_output_indexes.size());

    for (int i = 0; i < pipelined_input_indexes.size(); i++) {
      inputs[i] = ng_exec->create_input_tensor(pipelined_input_indexes[i],
                                               pipeline_depth);
    }

    for (int i = 0; i < pipelined_output_indexes.size(); i++) {
      outputs[i] = ng_exec->create_output_tensor(pipelined_output_indexes[i],
                                                 pipeline_depth);
    }

    return {inputs, outputs};
  }

  Status GetTensorFromBackend(shared_ptr<ngraph::runtime::Backend> ng_backend,
                              Tensor& tf_tensor,
                              shared_ptr<ngraph::runtime::Tensor>& ng_tensor) {
    ng::element::Type ng_element_type;
    TF_RETURN_IF_ERROR(
        TFDataTypeToNGraphElementType(tf_tensor.dtype(), &ng_element_type));
    ng::Shape ng_shape(tf_tensor.shape().dims());
    for (int j = 0; j < tf_tensor.shape().dims(); ++j) {
      ng_shape[j] = tf_tensor.shape().dim_size(j);
    }
    GetTensorFromBackend(ng_backend, ng_element_type, ng_shape, ng_tensor);
    cout<<"copy"<<endl;
    cout << "print tf ptr "<< &tf_tensor<<endl;
    cout << "print ng ptr "<< ng_tensor.get()<<endl;
    WriteNGTensor(ng_tensor, &tf_tensor);
    cout<<"copy"<<endl;

    return Status::OK();
  }

  Status GetTensorFromBackend(shared_ptr<ngraph::runtime::Backend> ng_backend,
                              ng::element::Type ng_element_type,
                              ng::Shape ng_shape,
                              shared_ptr<ngraph::runtime::Tensor>& ng_tensor) {
        cout<<"created"<<endl;

    ng_tensor = ng_backend->create_tensor(ng_element_type, ng_shape);
    cout << "print ng ptr "<< ng_tensor.get()<<endl;
    return Status::OK();
  }
};

TEST(NGraphExec, Axpy) {
  GraphDef gdef;
  // auto status = ReadTextProto(Env::Default(), "test_py.pbtxt",
  // &gdef);
  auto status =
      ReadTextProto(Env::Default(), "test_axpy_launchop.pbtxt", &gdef);
  // ReadTextProto(Env::Default(), "test_launch_op.pbtxt", &gdef);
  ASSERT_TRUE(status == Status::OK()) << "Can't read protobuf graph";

  Graph input_graph(OpRegistry::Global());

  GraphConstructorOptions opts;
  // Set the allow_internal_ops to true so that graphs with node names such as
  // _arg_Placeholder_1_0_1_0_arg are allowed. These op names are generated
  // during the graph rewrite passes and considered internal
  opts.allow_internal_ops = true;

  ASSERT_EQ(ConvertGraphDefToGraph(opts, gdef, &input_graph), Status::OK())
      << "Could not convert graphdef to graph";
  // Create the inputs for this graph
  Tensor x(DT_FLOAT, TensorShape({2, 3}));
  Tensor y(DT_FLOAT, TensorShape({2, 3}));

  std::vector<TensorShape> inputs;
  inputs.push_back(x.shape());
  inputs.push_back(y.shape());

  std::vector<const Tensor*> static_input_map(2, nullptr);

  shared_ptr<ng::Function> ng_function;
  ASSERT_EQ(Status::OK(),
            ngraph_bridge::Builder::TranslateGraph(inputs, static_input_map,
                                                   &input_graph, ng_function))
      << "Could not complete TranslateGraph successfully";

  // Create the nGraph backend
  auto backend = ng::runtime::Backend::create("CPU");

  // Allocate tensors for arguments a, b, c
  ng::Shape ng_shape_x(x.shape().dims());
  for (int i = 0; i < x.shape().dims(); ++i) {
    ng_shape_x[i] = x.shape().dim_size(i);
  }

  ng::Shape ng_shape_y(y.shape().dims());
  for (int i = 0; i < y.shape().dims(); ++i) {
    ng_shape_y[i] = y.shape().dim_size(i);
  }

  auto t_x = backend->create_tensor(ng::element::f32, ng_shape_x);
  float v_x[2][3] = {{1, 1, 1}, {1, 1, 1}};
  t_x->write(&v_x, 0, sizeof(v_x));

  auto t_y = backend->create_tensor(ng::element::f32, ng_shape_y);
  t_y->write(&v_x, 0, sizeof(v_x));

  // Allocate tensor for the result(s)
  vector<shared_ptr<ng::runtime::Tensor>> outputs;
  for (size_t i = 0; i < ng_function->get_output_size(); i++) {
    auto shape = ng_function->get_output_shape(i);
    auto elem_type = ng_function->get_output_element_type(i);
    auto t_result = backend->create_tensor(elem_type, shape);
    outputs.push_back(t_result);
  }

  // Execute the nGraph function.
  cout << "Calling nGraph function\n";
  auto exec = backend->compile(ng_function);
  exec->call(outputs, {t_x, t_y});

  for (size_t i = 0; i < ng_function->get_output_size(); i++) {
    DumpNGTensor<float>(cout, ng_function->get_output_op(i)->get_name(),
                        outputs[i]);
    cout << endl;
  }
  // Add the validation logic
  // TODO
}

TEST(NGraphExec, Axpy8bit) {
  GraphDef gdef;
  auto status =
      ReadTextProto(Env::Default(), "test_axpy_int8_launchop.pbtxt", &gdef);
  ASSERT_TRUE(status == Status::OK()) << "Can't read protobuf graph";

  Graph input_graph(OpRegistry::Global());

  GraphConstructorOptions opts;
  // Set the allow_internal_ops to true so that graphs with node names such as
  // _arg_Placeholder_1_0_1_0_arg are allowed. These op names are generated
  // during the graph rewrite passes and considered internal
  opts.allow_internal_ops = true;

  ASSERT_EQ(ConvertGraphDefToGraph(opts, gdef, &input_graph), Status::OK())
      << "Could not convert graphdef to graph";
  // Create the inputs for this graph
  Tensor x(DT_INT8, TensorShape({2, 2}));
  Tensor y(DT_INT8, TensorShape({2, 2}));

  std::vector<TensorShape> inputs;
  inputs.push_back(x.shape());
  inputs.push_back(y.shape());

  std::vector<const Tensor*> static_input_map(2, nullptr);

  shared_ptr<ng::Function> ng_function;
  ASSERT_EQ(Status::OK(),
            ngraph_bridge::Builder::TranslateGraph(inputs, static_input_map,
                                                   &input_graph, ng_function))
      << "Could not complete TranslateGraph successfully";

  // Create the nGraph backend
  auto backend = ng::runtime::Backend::create("CPU");

  // Allocate tensors for arguments a, b, c
  ng::Shape ng_shape_x(x.shape().dims());
  for (int i = 0; i < x.shape().dims(); ++i) {
    ng_shape_x[i] = x.shape().dim_size(i);
  }

  ng::Shape ng_shape_y(y.shape().dims());
  for (int i = 0; i < y.shape().dims(); ++i) {
    ng_shape_y[i] = y.shape().dim_size(i);
  }

  auto t_x = backend->create_tensor(ng::element::i8, ng_shape_x);
  int8 v_x[2][2] = {{1, 1}, {1, 1}};
  t_x->write(&v_x, 0, sizeof(v_x));

  auto t_y = backend->create_tensor(ng::element::i8, ng_shape_y);
  t_y->write(&v_x, 0, sizeof(v_x));

  // Allocate tensor for the result(s)
  vector<shared_ptr<ng::runtime::Tensor>> outputs;
  for (size_t i = 0; i < ng_function->get_output_size(); i++) {
    auto shape = ng_function->get_output_shape(i);
    auto elem_type = ng_function->get_output_element_type(i);
    auto t_result = backend->create_tensor(elem_type, shape);
    outputs.push_back(t_result);
  }

  // Execute the nGraph function.
  cout << "Calling nGraph function\n";
  auto exec = backend->compile(ng_function);
  exec->call(outputs, {t_x, t_y});

  for (size_t i = 0; i < ng_function->get_output_size(); i++) {
    DumpNGTensor<int8>(cout, ng_function->get_output_op(i)->get_name(),
                       outputs[i]);
    cout << endl;
  }
  // Add the validation logic
  // TODO
}

TEST_F(NGraphExecTest, MixedTensors) {
  GraphDef gdef;
  ASSERT_OK(ReadTextProto(Env::Default(), "test_axpy_launchop.pbtxt", &gdef))
      << "Can't read protobuf graph";

  Graph input_graph(OpRegistry::Global());

  GraphConstructorOptions opts;
  // Set the allow_internal_ops to true so that graphs with node names such as
  // _arg_Placeholder_1_0_1_0_arg are allowed. These op names are generated
  // during the graph rewrite passes and considered internal
  opts.allow_internal_ops = true;

  ASSERT_OK(ConvertGraphDefToGraph(opts, gdef, &input_graph))
      << "Could not convert graphdef to graph";

  // Create the inputs for this graph
  DataType tf_dt = DT_FLOAT;
  TensorShape tf_shape = TensorShape({2, 3});
  Tensor x(tf_dt, tf_shape);
  Tensor y(tf_dt, tf_shape);
  AssignInputValues(x, 1.0f);
  AssignInputValues(y, 1.0f);
  std::vector<Tensor> tf_inputs = {x, y};

  std::vector<TensorShape> tf_input_shapes;
  tf_input_shapes.push_back(x.shape());
  tf_input_shapes.push_back(y.shape());

  // Translate the Graph: Create ng_function
  std::vector<const Tensor*> static_input_map(2, nullptr);
  shared_ptr<ng::Function> ng_function;
  ASSERT_EQ(Status::OK(),
            ngraph_bridge::Builder::TranslateGraph(
                tf_input_shapes, static_input_map, &input_graph, ng_function))
      << "Could not complete TranslateGraph successfully";

  // Create the nGraph backend
  string backend_name = "INTERPRETER";
  OverrideBackendFromEnv(&backend_name);
  auto backend = ng::runtime::Backend::create(backend_name);
  NGRAPH_VLOG(0) << "NGraph using backend " << backend_name << endl;

  // check if the backend executable can create tensors
  ASSERT_TRUE(backend->executable_can_create_tensors())
      << "Backend Executable cannot create tensors";

  // Compile the nGraph function.
  auto exec = backend->compile(ng_function);

  // Allocate ng tensors for inputs
  vector<shared_ptr<ng::runtime::Tensor>> ng_inputs;

  for (int i = 0; i < 2; ++i) {
    shared_ptr<ng::runtime::Tensor> ng_input;
    if (i % 2 == 0) {
      ng_input = exec->create_input_tensor(i);
    } else {
      ng::element::Type ng_element_type;
      ASSERT_OK(TFDataTypeToNGraphElementType(tf_inputs[i].dtype(),
                                              &ng_element_type));
      ng::Shape ng_shape(tf_inputs[i].shape().dims());
      for (int j = 0; j < tf_inputs[i].shape().dims(); ++j) {
        ng_shape[j] = tf_inputs[i].shape().dim_size(j);
      }
      ng_input = backend->create_tensor(ng_element_type, ng_shape);
    }
    void* src_ptr = DMAHelper::base(&tf_inputs[i]);
    ng_input->write(src_ptr, 0, tf_inputs[i].TotalBytes());
    ng_inputs.push_back(ng_input);
  }

  // Allocate tensor for the result(s)
  vector<shared_ptr<ng::runtime::Tensor>> ng_outputs;

  for (size_t i = 0; i < ng_function->get_output_size(); i++) {
    shared_ptr<ng::runtime::Tensor> ng_output;
    if (i % 2 == 0) {
      ng_output = exec->create_output_tensor(i);
    } else {
      auto shape = ng_function->get_output_shape(i);
      auto elem_type = ng_function->get_output_element_type(i);
      ng_output = backend->create_tensor(elem_type, shape);
    }
    ng_outputs.push_back(ng_output);
  }

  // Execute the nGraph function.
  exec->call(ng_outputs, ng_inputs);

  // Actual Outputs
  // Allocating TF Tensors and reading into them to compare the outputs
  vector<Tensor> actual_outputs;

  for (size_t i = 0; i < ng_function->get_output_size(); i++) {
    // Convert to tf tensor
    Tensor output_tensor(tf_dt, tf_shape);
    void* dst_ptr = DMAHelper::base(&output_tensor);
    ng_outputs[i]->read(dst_ptr, 0, output_tensor.TotalBytes());
    actual_outputs.push_back(output_tensor);
  }

  // Expected output
  Tensor output1(DT_FLOAT, TensorShape({2, 3}));
  Tensor output2(DT_FLOAT, TensorShape({2, 3}));
  AssignInputValues(output1, 6.0f);
  AssignInputValues(output2, 5.0f);
  vector<Tensor> expected_outputs = {output1, output2};

  // Comparing
  Compare(expected_outputs, actual_outputs);
}

TEST_F(NGraphExecTest, MixedTensorsPipelined) {
  Graph input_graph(OpRegistry::Global());
  ASSERT_OK(LoadGraph("test_graph_1.pbtxt", &input_graph));

  // Create the inputs for this graph
  int num_inputs = 3;
  DataType tf_dt = DT_FLOAT;
  TensorShape tf_shape = TensorShape({2, 3});
  Tensor x(tf_dt, tf_shape);
  Tensor y(tf_dt, tf_shape);
  Tensor z(tf_dt, tf_shape);
  AssignInputValues(x, 1.0f);
  AssignInputValues(y, 2.0f);
  AssignInputValues(z, 3.0f);

  std::vector<Tensor> tf_inputs = {x, y, z};
  std::vector<TensorShape> tf_input_shapes = {x.shape(), y.shape(), z.shape()};

  // Translate the Graph: Create ng_function
  shared_ptr<ngraph::Function> ng_function;
  std::vector<const Tensor*> static_input_map(num_inputs, nullptr);
  ASSERT_OK(ngraph_bridge::Builder::TranslateGraph(
      tf_input_shapes, static_input_map, &input_graph, ng_function))
      << "Could not complete TranslateGraph successfully";

  // Create the nGraph backend
  string backend_name = "INTERPRETER";
  OverrideBackendFromEnv(&backend_name);
  auto backend = ng::runtime::Backend::create(backend_name);
  NGRAPH_VLOG(0) << "NGraph using backend " << backend_name << endl;

  // check if the backend executable can create tensors
  ASSERT_TRUE(backend->executable_can_create_tensors())
      << "Backend Executable cannot create tensors";

  // Compile the nGraph function.
  shared_ptr<ngraph::runtime::Executable> ng_exec;
  ng_exec = backend->compile(ng_function);

  int num_outputs = ng_function->get_output_size();
  int pipeline_depth = 2;

  // Allocate ng tensors for inputs and outputs
  vector<shared_ptr<ng::runtime::Tensor>> ng_inputs(num_inputs, nullptr);
  vector<shared_ptr<ng::runtime::Tensor>> ng_outputs(num_outputs, nullptr);

  // Lets assume inputs and outputs 0 and 2 are pipelined
  vector<int> pipelined_input_indexes = {0, 2};
  vector<int> pipelined_output_indexes = {0, 2};
  vector<int> non_pipelined_input_indexes = {1};
  vector<int> non_pipelined_output_indexes = {1};

  auto inp_out =
      CreatePipelinedTensors(ng_exec, pipelined_input_indexes,
                             pipelined_output_indexes, pipeline_depth);
  PipelinedTensorMatrix pipelined_inputs = get<0>(inp_out);
  PipelinedTensorMatrix pipelined_outputs = get<1>(inp_out);
  cout<<"Created Pipelined Tensor "<<endl;
  for (int itr = 0; itr < pipeline_depth; itr++) {
    // Prepare Inputs
    // Get Backend Tensors
    for (auto index : non_pipelined_input_indexes) {
      cout<<"Getting Backend Tensor "<<index<<endl;
  
      GetTensorFromBackend(backend, tf_inputs[index], ng_inputs[index]);
    }
    // Get Pipelined Tensors
    for (auto index : pipelined_input_indexes) {
      int count=0;
      ng_inputs[index] = pipelined_inputs[count++][itr];
      WriteNGTensor(ng_inputs[index], &tf_inputs[index]);
      cout<<"Created Pipelined Tensor "<<index <<endl;
  
    }

    // Prepare Outputs
    // Get Backend Tensors
    for (auto index : non_pipelined_output_indexes) {
      cout<<"Getting Backend Tensor op "<<index<<endl;
  
      auto shape = ng_function->get_output_shape(index);
      auto elem_type = ng_function->get_output_element_type(index);
      GetTensorFromBackend(backend, elem_type, shape, ng_outputs[index]);
    }
    for (auto index : pipelined_output_indexes) {
      cout<<"Created Pipelined op Tensor "<<index <<endl;
      int count=0;
      ng_outputs[index] = pipelined_outputs[count++][itr];
      cout<<"Created Pipelined op Tensor "<<index <<endl;
    }

    for(auto ptr: ng_inputs){
      cout<<ptr<<endl;
    }

    for(auto ptr: ng_outputs){
      cout<<ptr<<endl;
    }
    // call
    ng_exec->call(ng_outputs, ng_inputs);

    // compare

  for (size_t i = 0; i < num_outputs; i++) {
    // Convert to tf tensor
    Tensor output_tensor(tf_dt, tf_shape);
    void* dst_ptr = DMAHelper::base(&output_tensor);
    ng_outputs[i]->read(dst_ptr, 0, output_tensor.TotalBytes());
    //actual_outputs.push_back(output_tensor);
    cout<<output_tensor.DebugString()<<endl;
  }

    // clear
    break;
  }

  
}



}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow
