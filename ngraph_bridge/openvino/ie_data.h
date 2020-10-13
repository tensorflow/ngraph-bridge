
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


#ifndef IE_DATA_H_
#define IE_DATA_H_

#include <ie_core.hpp>
#include <memory>
#include <string>
#include <vector>


class IE_Data {
 public:
  IE_Data(const void *data_pointer,
          InferenceEngine::Precision precision,
          InferenceEngine::Layout layout,
          InferenceEngine::SizeVector shape,
          size_t byte_size,
	  std::string name);
  IE_Data(const void *data_pointer,
          InferenceEngine::Precision precision,
          InferenceEngine::Layout layout,
          InferenceEngine::SizeVector shape,
          size_t byte_size);

  const void * get_data_pointer() const;
  InferenceEngine::Precision get_precision() const;
  InferenceEngine::Layout get_layout() const;
  InferenceEngine::SizeVector get_shape() const;
  size_t get_byte_size() const;
  void set_name(std::string name);
  std::string get_name() const;
  bool has_name() const;

  void write(const void* src, size_t bytes);
  void read(void* dst, size_t bytes) const;

  const void* get_data_ptr() const;

 private:
  const void *m_data_pointer;
  InferenceEngine::Precision m_precision;
  InferenceEngine::Layout m_layout;
  InferenceEngine::SizeVector m_shape;
  size_t m_byte_size;
  std::string m_name;
};


#endif  // IE_DATA_H_
