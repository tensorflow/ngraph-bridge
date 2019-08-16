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

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"

#include "logging/tf_graph_writer.h"
#include "ngraph_bridge/ngraph_partial_shapes.h"
#include "ngraph_bridge/ngraph_utils.h"
#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

#define ASSERT_OK(x) ASSERT_EQ((x), ::tensorflow::Status::OK());
#define ASSERT_NOT_OK(x) ASSERT_NE((x), ::tensorflow::Status::OK());

<<<<<<< HEAD
// The result of concretize would be {2, 3}
TEST(PartialShapes, ValidConcretize1) {
    PartialShape p1({2, -1});
    PartialShape p2({-1, 3});
    p1.concretize(p2);
    ASSERT_EQ(p1.is_valid(), true);
}

// The result of concretize would be {-1, -1}
TEST(PartialShapes, ValidConcretize2) {
    PartialShape p1({-1, -1});
    PartialShape p2({-1, -1});
    p1.concretize(p2);
    ASSERT_EQ(p1.is_valid(), true);
=======
TEST(PartialShapes, Concretize) {
  PartialShape p1({2, -1});
  PartialShape p2({-1, 3});
  cout << p1.is_valid() << "\n";
  p1.concretize(p2);
  bool result = p1.is_valid();
  cout << "result " << result;
>>>>>>> 26ab585cbd74e8df336aac6f11bbf2a7c2799c3a
}

// The result of concretize would be {}
TEST(PartialShapes, ValidConcretize3) {
    PartialShape p1(vector<int>{});
    PartialShape p2(vector<int>{});
    p1.concretize(p2);
    ASSERT_EQ(p1.is_valid(), true);
}

// This would result in an invalid case and should fail because
// p1[0] = 2 does not match p2[0] = 3
TEST(PartialShapes, InvalidConcretize1) {
    PartialShape p1({2, -1});
    PartialShape p2({3, 3});
    p1.concretize(p2);
    ASSERT_EQ(p1.is_valid(), false);
}

// This would result in an invalid case and should fail because
// the ranks are not same
TEST(PartialShapes, InvalidConcretize2) {
    PartialShape p1({2, -1, -1});
    PartialShape p2({3, 3});
    p1.concretize(p2);
    ASSERT_EQ(p1.is_valid(), false);
}

// The result of concretize would be {3, 3}, which is a concrete shape
TEST(PartialShapes, IsConcrete1) {
    PartialShape p1({-1, -1});
    PartialShape p2({3, 3});
    p1.concretize(p2);
    ASSERT_EQ(p1.is_concrete(), true);
}

// The result of concretize would be {3, -1}, which is not a concrete shape
TEST(PartialShapes, IsConcrete2) {
    PartialShape p1({-1, -1});
    PartialShape p2({3, -1});
    p1.concretize(p2);
    ASSERT_EQ(p1.is_concrete(), false);
}

// The result of concretize would be {}, which is a concrete shape
TEST(PartialShapes, IsConcrete3) {
    PartialShape p1(vector<int>{});
    PartialShape p2(vector<int>{});
    p1.concretize(p2);
    ASSERT_EQ(p1.is_concrete(), true);
}

// Test default constructor
TEST(PartialShapes, DefaultConstructor) {
    PartialShape p1;
    ASSERT_EQ(p1.is_valid(), false);
}

TEST(PartialShapes, Constructor) {
    PartialShape p1({-2, 1});
    ASSERT_EQ(p1.is_valid(), false);
}


}  // namespace testing

}  // namespace ngraph_bridge

}  // namespace tensorflow
