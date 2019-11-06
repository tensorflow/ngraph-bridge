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


#include "test/opexecuter.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

// Test op: ResizeBilinear
TEST(ImageOps, ResizeBilinear) {
  Scope root = Scope::NewRootScope();
  // [batch, height, width, channels]
  Tensor images(DT_FLOAT, TensorShape({4,64,64,3}));
  AssignInputValuesRandom(images);

  // Todo: test by changing align_corners

  // new_height, new_width
  Tensor size(DT_INT32, TensorShape({2}));
  vector<int> new_dims = {93, 27}; // TODO loop and do multiple sizes, larger and smaller than original
  AssignInputValues(size, new_dims);

  vector<int> static_input_indexes = {};
  auto R = ops::ResizeBilinear(root, images, size);
  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "ResizeBilinear", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}


}
}
}