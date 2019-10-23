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
#include <atomic>
#include <memory>
#include "absl/synchronization/barrier.h"
#include "gtest/gtest.h"
#include "test/test_utilities.h"

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/public/session.h"

#include "ngraph_bridge/ngraph_backend_manager.h"
#include "ngraph_bridge/ngraph_builder.h"
#include "ngraph_bridge/ngraph_data_cache.h"
#include "ngraph_bridge/version.h"

using namespace std;
namespace ng = ngraph;
namespace tf = tensorflow;

namespace tensorflow {
namespace ngraph_bridge {
namespace testing {

class NGraphDataCacheTest : public ::testing::Test {
 protected:
  NgraphDataCache<std::string, int> m_ng_data_cache{3};
  int num_threads = 2;
  absl::Barrier* barrier_ = new absl::Barrier(num_threads);
  std::atomic<int> create_count{0};
  bool item_evicted = false;

  std::pair<Status, int> CreateItem() {
    create_count++;
    if (barrier_->Block()) {
      delete barrier_;
    }
    return std::make_pair(Status::OK(), 3);
  }

  std::pair<Status, int> CreateItemNoBarrier() {
    return std::make_pair(Status::OK(), 3);
  }

  void DestroyItem(int i) { item_evicted = true; }
};

TEST_F(NGraphDataCacheTest, SameKeyMultiThread) {
  auto worker = [&](size_t thread_id) {
    auto create_item = std::bind(
        &NGraphDataCacheTest_SameKeyMultiThread_Test::CreateItem, this);
    m_ng_data_cache.LookUpOrCreate("abc", create_item, [](int) {});
    m_ng_data_cache.LookUpOrCreate("abc", create_item);
  };

  std::thread thread0(worker, 0);
  std::thread thread1(worker, 1);

  thread0.join();
  thread1.join();
  ASSERT_EQ(create_count, 2);
  ASSERT_EQ(m_ng_data_cache.m_ng_items_map.size(), 1);
}

TEST_F(NGraphDataCacheTest, TestItemEviction) {
  auto create_item = std::bind(
      &NGraphDataCacheTest_TestItemEviction_Test::CreateItemNoBarrier, this);
  auto destroy_item =
      std::bind(&NGraphDataCacheTest_TestItemEviction_Test::DestroyItem, this,
                std::placeholders::_1);

  m_ng_data_cache.LookUpOrCreate("abc", create_item);
  m_ng_data_cache.LookUpOrCreate("def", create_item, destroy_item);
  m_ng_data_cache.LookUpOrCreate("efg", create_item, destroy_item);
  ASSERT_EQ(item_evicted, false);
  m_ng_data_cache.LookUpOrCreate("hij", create_item, destroy_item);
  ASSERT_EQ(item_evicted, true);
  m_ng_data_cache.LookUpOrCreate("klm", create_item);
}
}
}
}
