/*******************************************************************************
 * Copyright 2017-2020 Intel Corporation
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

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

#include "ngraph_bridge/backend_manager.h"
#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {
namespace ngraph_bridge {
namespace testing {

#define ASSERT_NOT_OK(x) ASSERT_NE((x), ::tensorflow::Status::OK());

/*
These tests test the Backend Handling by the bridge.
*/

// Test SetBackendAPI
TEST(BackendManager, SetBackend) {
  auto env_map = StoreEnv({"NGRAPH_TF_BACKEND"});

  EXPECT_NO_THROW(BackendManager::SetBackend("CPU"));
  auto backend = BackendManager::GetBackend();
  ASSERT_EQ(backend->name(), "CPU");
  EXPECT_ANY_THROW(BackendManager::SetBackend("temp"));

  // Clean Up
  // If NGRAPH_TF_BACKEND was set, set it back
  RestoreEnv(env_map);
}

// Test GetBackend API
TEST(BackendManager, GetBackend) {
  auto env_map = StoreEnv({"NGRAPH_TF_BACKEND"});

  EXPECT_NO_THROW(BackendManager::SetBackend("CPU"));
  auto backend = BackendManager::GetBackend();
  ASSERT_EQ(backend->name(), "CPU");

  // expected ERROR
  SetBackendUsingEnvVar("DUMMY");
  backend = BackendManager::GetBackend();
  ASSERT_EQ(backend->name(), "CPU");

  // set env variable to ""
  SetBackendUsingEnvVar("");
  backend = BackendManager::GetBackend();
  ASSERT_EQ(backend->name(), "CPU");

  // set backend to dummy and env variable to CPU
  // expected CPU
  EXPECT_ANY_THROW(BackendManager::SetBackend("DUMMY"));
  SetBackendUsingEnvVar("CPU");
  backend = BackendManager::GetBackend();
  ASSERT_EQ(backend->name(), "CPU");

  // unset env variable
  // expected interpreter
  UnsetBackendUsingEnvVar();
  backend = BackendManager::GetBackend();
  ASSERT_EQ(backend->name(), "CPU");

  // Clean up
  UnsetBackendUsingEnvVar();
  EXPECT_NO_THROW(BackendManager::SetBackend("CPU"));
  // restore
  // If NGRAPH_TF_BACKEND was set, set it back
  RestoreEnv(env_map);
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow
