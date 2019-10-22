// Executor Cache.h

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

#ifndef NGRAPH_DATA_CACHE_H_
#define NGRAPH_DATA_CACHE_H_
#pragma once

#include <mutex>
#include <ostream>
#include <vector>
#include "absl/synchronization/mutex.h"

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/graph.h"

#include "logging/ngraph_log.h"
#include "ngraph/ngraph.hpp"

#include "ngraph_bridge/ngraph_freshness_tracker.h"
#include "ngraph_bridge/ngraph_pipelined_tensors.h"

namespace tensorflow {

namespace ngraph_bridge {

// Forward declaration for friend class
namespace testing {
class NGraphDataCacheTest_SameKeyMultiThread_Test;
class NGraphDataCacheTest_DiffKeyMultiThread_Test;
}

template <typename T>
class NgraphDataCache {
 public:
  NgraphDataCache(int depth);
  ~NgraphDataCache();

  // This method performs lookup in the cache for requested key, if not found
  // it will create item, put it in the cache and returns item and stattus
  std::pair<Status, T> LookUpOrCreate(
      std::string key, std::function<std::pair<Status, T>()> create_item,
      std::function<void(T)> callback_destroy_item);
  std::pair<Status, T> LookUpOrCreate(
      std::string key, std::function<std::pair<Status, T>()> create_item);

 private:
  std::unordered_map<std::string, T> m_ng_items_map;
  std::list<std::string> m_lru;
  int m_depth;
  absl::Mutex m_mutex;
  ;

  friend class tensorflow::ngraph_bridge::testing::
      NGraphDataCacheTest_SameKeyMultiThread_Test;
  friend class tensorflow::ngraph_bridge::testing::
      NGraphDataCacheTest_DiffKeyMultiThread_Test;
};

template <typename T>
NgraphDataCache<T>::NgraphDataCache(int depth) : m_depth(depth) {}

template <typename T>
NgraphDataCache<T>::~NgraphDataCache() {
  m_ng_items_map.clear();
}

template <typename T>
std::pair<Status, T> NgraphDataCache<T>::LookUpOrCreate(
    std::string key, std::function<std::pair<Status, T>()> callback_create_item,
    std::function<void(T)> callback_destroy_item) {
  bool found_in_cache;
  // look up in the cache
  {
    absl::MutexLock lock(&m_mutex);
    auto it = m_ng_items_map.find(key);
    found_in_cache = (it != m_ng_items_map.end());
    if (found_in_cache) {
      return std::make_pair(Status::OK(), m_ng_items_map.at(key));
    }
  }
  // Item not found in cache, create item
  T item;
  auto status_item_pair = callback_create_item();

  // If item is successfully created we will place in the cache.
  if (status_item_pair.first == Status::OK()) {
    item = status_item_pair.second;
    T item_to_evict;
    bool need_to_evict = false;
    // lock begins
    {
      absl::MutexLock lock(&m_mutex);
      // Remove item if cache is full
      if (m_ng_items_map.size() == m_depth) {
        auto key_to_evict = m_lru.back();
        item_to_evict = m_ng_items_map.at(key_to_evict);
        need_to_evict = true;
        m_ng_items_map.erase(key_to_evict);
        m_lru.pop_back();
      }
      // Add item to cache
      auto it = m_ng_items_map.emplace(key, item);
      if (it.second == true) {
        m_lru.push_front(key);
      }
    }  // lock ends here.
    if (need_to_evict) {
      callback_destroy_item(item_to_evict);
    }

    return std::make_pair(Status::OK(), item);
  }

  return status_item_pair;
}

template <typename T>
std::pair<Status, T> NgraphDataCache<T>::LookUpOrCreate(
    std::string key,
    std::function<std::pair<Status, T>()> callback_create_item) {
  return LookUpOrCreate(key, callback_create_item, [](T) {});
}
}
}
#endif  // NGRAPH_DATA_CACHE_H_
