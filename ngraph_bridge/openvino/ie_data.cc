
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

#include "ngraph_bridge/openvino/ie_data.h"
#include <iostream>


IE_Data::IE_Data(const void *data_pointer,
                 InferenceEngine::Precision precision,
                 InferenceEngine::Layout layout,
                 InferenceEngine::SizeVector shape,
                 size_t byte_size,
		 std::string name)
    : m_data_pointer(data_pointer), m_precision(precision), m_layout(layout), m_shape(shape), m_byte_size(byte_size), m_name(name) {
}

IE_Data::IE_Data(const void *data_pointer,
                 InferenceEngine::Precision precision,
                 InferenceEngine::Layout layout,
                 InferenceEngine::SizeVector shape,
                 size_t byte_size)
    : m_data_pointer(data_pointer), m_precision(precision), m_layout(layout), m_shape(shape), m_byte_size(byte_size), m_name("") {
}

const void * IE_Data::get_data_pointer() const {
    return m_data_pointer;
}

InferenceEngine::Precision IE_Data::get_precision() const {
    return m_precision;
}

InferenceEngine::Layout IE_Data::get_layout() const {
    return m_layout;
}

InferenceEngine::SizeVector IE_Data::get_shape() const {
    return m_shape;
}

size_t IE_Data::get_byte_size() const {
    if (m_data_pointer == nullptr)
        return 0;
    else
        return m_byte_size;
}

void IE_Data::set_name(std::string name) {
    m_name = name;
}

std::string IE_Data::get_name() const {
    return m_name;
}

bool IE_Data::has_name() const {
    return !(m_name.empty());
}

void IE_Data::write(const void* src, size_t bytes) {
    std::memcpy((void*)m_data_pointer, src, bytes);
}

void IE_Data::read(void* dst, size_t bytes) const {
    std::memcpy(dst, m_data_pointer, bytes);
}

const void* IE_Data::get_data_ptr() const {
    return (const void*)m_data_pointer;
};
