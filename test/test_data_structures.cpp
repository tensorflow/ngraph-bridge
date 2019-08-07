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

#include "ngraph_bridge/ngraph_pipelined_tensors.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

TEST(IndexLibrary, SingleThreadTest1) {
  IndexLibrary idx_lib{3};
  set<int> expected{0, 1, 2};

  int i0 = idx_lib.get_index();
  ASSERT_TRUE(expected.find(i0) != expected.end());
  expected.erase(i0);  // 2 elements left, i0 checked out

  int i1 = idx_lib.get_index();
  ASSERT_TRUE(expected.find(i1) != expected.end());
  expected.erase(i1);  // 1 elements left, i0, i1 checked out

  idx_lib.return_index(i0);
  expected.insert(i0);  // 2 elements left, i1 checked out

  int i2 = idx_lib.get_index();
  ASSERT_TRUE(expected.find(i2) != expected.end());
  expected.erase(i2);  // 1 elements left, i1, i2 checked out

  int i3 = idx_lib.get_index();
  ASSERT_TRUE(expected.find(i3) != expected.end());
  expected.erase(i3);  // 0 elements left, i1, i2, i3 checked out

  int i4 = idx_lib.get_index();
  ASSERT_EQ(i4, -1)
      << "Expected index library to be empty, hence get_index should return -1";

  // Try to return an invalid index
  ASSERT_THROW(idx_lib.return_index(50), std::runtime_error);

  idx_lib.return_index(i1);
  expected.insert(i1);  // 1 elements left, i2, i3 checked out

  // Try to return an index that is already checkedin/returned
  ASSERT_THROW(idx_lib.return_index(i1), std::runtime_error);
}

TEST(IndexLibrary, SingleThreadTest2) {
  IndexLibrary idx_lib{0};

  // Since it is an empty library it will always return -1
  ASSERT_EQ(idx_lib.get_index(), -1);
}

// TODO write 2 thread UT for IndexLibrary
}
}
}