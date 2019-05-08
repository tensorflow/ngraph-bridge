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
#include "opexecuter.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {
// TODO add DISABLED_
TEST(ImageOps, DISABLED_CombinedNMS) {
  Scope root = Scope::NewRootScope();
  int batch_size = 10;
  int num_anchors = 2;
  int q = 1;
  int num_classes = 20;

  Tensor boxes(DT_FLOAT, TensorShape({batch_size, num_anchors, q, 4}));
  AssignInputValuesRandom<float>(boxes, 1.0f,
                                 20.0f);  // TODO: Can it have negative numbers?

  Tensor scores(DT_FLOAT, TensorShape({batch_size, num_anchors, num_classes}));
  AssignInputValuesRandom<float>(scores, 0.0f, 1.0f);

  Tensor max_output_size_per_class(DT_INT32, TensorShape({}));
  AssignInputValues<int>(max_output_size_per_class, {3});

  Tensor max_total_size(DT_INT32, TensorShape({}));
  AssignInputValues<int>(max_total_size, {4});

  Tensor iou_threshold(DT_FLOAT, TensorShape({}));
  AssignInputValues<float>(iou_threshold, {0.2f});

  Tensor score_threshold(DT_FLOAT, TensorShape({}));
  AssignInputValues<float>(score_threshold, {0.3f});

  vector<int> static_input_indexes = {2, 3, 4, 5};
  auto R = ops::CombinedNonMaxSuppression(
      root, boxes, scores, max_output_size_per_class, max_total_size,
      iou_threshold, score_threshold);

  vector<DataType> output_datatypes = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_INT32};
  std::vector<Output> sess_run_fetchoutputs = {
      R.nmsed_boxes, R.nmsed_scores, R.nmsed_classes, R.valid_detections};
  OpExecuter opexecuter(root, "CombinedNonMaxSuppression", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);

  opexecuter.RunTest();

  // TODO what of attributes: pad_per_class, clip_boxes

}  // end of test op CombinedNonMaxSuppression
}
}
}
