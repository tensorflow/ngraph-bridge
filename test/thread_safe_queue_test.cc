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
#include <cstdlib>
#include <thread>
#include <utility>

#include "gtest/gtest.h"
#include "ngraph_bridge/thread_safe_queue.h"
#include "tensorflow/core/public/session.h"

#include "absl/time/clock.h"
#include "absl/time/time.h"

using namespace std;

namespace tensorflow {
namespace ngraph_bridge {

namespace testing {
TEST(ThreadSafeQueue, Simple) {
  ThreadSafeQueue<Session> queue;
  typedef enum {
    INIT = 0,
    WAITING_FOR_ITEM,
    READY_TO_WAIT,
    GOT_ITEM,
  } CONSUMER_STATE;

  atomic<CONSUMER_STATE> consumer_state{INIT};
  atomic<bool> consumer_do_wait{true};
  atomic<int> item_count{0};

  bool run_test{true};

  // Create two threads
  auto consumer = [&]() {
    while (run_test) {
      while (consumer_do_wait) {
        absl::SleepFor(absl::Milliseconds(500));
      }
      consumer_state = WAITING_FOR_ITEM;
      cout << "Waiting" << endl;
      queue.GetNextAvailable();
      cout << "Got Item: " << item_count << endl;
      item_count++;
      consumer_state = GOT_ITEM;
      consumer_do_wait = true;
      cout << "Starting waiting" << endl;
      consumer_state = READY_TO_WAIT;
    }
  };

  std::thread thread0(consumer);

  // Ensure that the consumer is in waiting state
  ASSERT_TRUE(consumer_do_wait);

  consumer_do_wait = false;
  while (consumer_state != WAITING_FOR_ITEM) {
    absl::SleepFor(absl::Milliseconds(10));
  }

  cout << "Now adding an item\n";
  queue.Add(nullptr);
  // Wait until the consumer has a chance to move forward
  while (consumer_state != READY_TO_WAIT) {
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_EQ(item_count, 1);

  // THe consumer is now waiting again until consumer_do_wait is signaled
  // Add two more items
  cout << "Now adding two items\n";

  queue.Add(nullptr);
  queue.Add(nullptr);

  ASSERT_EQ(consumer_state, READY_TO_WAIT);
  cout << "Checking ...\n";

  // Now signal the consumer to stop
  // while(consumer_state != READY_TO_WAIT) {
  //     absl::SleepFor(absl::Milliseconds(10));
  // }
  consumer_do_wait = false;
  while (item_count != 2) {
    absl::SleepFor(absl::Milliseconds(10));
  }

  cout << "Consumer got item 2\n";

  // We set this flag to false so that as soon as the last
  // item is pulled out of the queue, the consumer thread terminates
  run_test = false;
  while (item_count != 3) {
    absl::SleepFor(absl::Milliseconds(10));
    consumer_do_wait = false;
  }

  thread0.join();
}
}

}  // namespace ngraph_bridge
}  // namespace tensorflow
