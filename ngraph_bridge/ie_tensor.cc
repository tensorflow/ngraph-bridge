//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <cstring>
#include <memory>
#include <utility>

#include "ngraph/ngraph.hpp"

#include "ie_layouts.h"
#include "ie_precision.hpp"
#include "ie_tensor.h"

using namespace ngraph;
using namespace std;

namespace tensorflow {
namespace ngraph_bridge {

static InferenceEngine::Precision toPrecision(
    const element::Type& element_type) {
  switch (element_type) {
    case element::Type_t::f32:
      return InferenceEngine::Precision::FP32;
    case element::Type_t::u8:
      return InferenceEngine::Precision::U8;
    case element::Type_t::i8:
      return InferenceEngine::Precision::I8;
    case element::Type_t::u16:
      return InferenceEngine::Precision::U16;
    case element::Type_t::i16:
      return InferenceEngine::Precision::I16;
    case element::Type_t::i32:
      return InferenceEngine::Precision::I32;
    case element::Type_t::u64:
      return InferenceEngine::Precision::U64;
    case element::Type_t::i64:
      return InferenceEngine::Precision::I64;
    case element::Type_t::boolean:
      return InferenceEngine::Precision::BOOL;
    default:
      THROW_IE_EXCEPTION << "Can't convert type " << element_type
                         << " to IE precision!";
  }
}

static const element::Type fromPrecision(
    const InferenceEngine::Precision precision) {
  switch (precision) {
    case InferenceEngine::Precision::FP32:
      return element::Type_t::f32;
    case InferenceEngine::Precision::U8:
      return element::Type_t::u8;
    case InferenceEngine::Precision::I8:
      return element::Type_t::i8;
    case InferenceEngine::Precision::U16:
      return element::Type_t::u16;
    case InferenceEngine::Precision::I16:
      return element::Type_t::i16;
    case InferenceEngine::Precision::I32:
      return element::Type_t::i32;
    case InferenceEngine::Precision::U64:
      return element::Type_t::u64;
    case InferenceEngine::Precision::I64:
      return element::Type_t::i64;
    case InferenceEngine::Precision::BOOL:
      return element::Type_t::boolean;
    default:
      THROW_IE_EXCEPTION << "Can't convert IE precision " << precision
                         << " to nGraph type!";
  }
}

IETensor::IETensor(const element::Type& element_type, const Shape& shape_,
                   void* memory_pointer)
    : runtime::Tensor(
          make_shared<descriptor::Tensor>(element_type, shape_, "")) {
  InferenceEngine::SizeVector shape = shape_;
  InferenceEngine::Precision precision = toPrecision(element_type);
  InferenceEngine::Layout layout =
      InferenceEngine::TensorDesc::getLayoutByDims(shape);

  auto size = shape_size(shape_) * element_type.size();

  m_data_pointer = memory_pointer;
  m_precision = precision;
  m_layout = layout;
  m_shape = shape;
  m_byte_size = size;
  m_name = "";
}

IETensor::IETensor(const element::Type& element_type, const Shape& shape)
    : IETensor(element_type, shape, nullptr) {}

IETensor::IETensor(const element::Type& element_type, const PartialShape& shape)
    : runtime::Tensor(
          make_shared<descriptor::Tensor>(element_type, shape, "")) {
  throw runtime_error("partial shapes not supported.");
}

IETensor::IETensor(const void* data_pointer,
                   InferenceEngine::Precision precision,
                   InferenceEngine::Layout layout,
                   InferenceEngine::SizeVector shape, size_t byte_size,
                   std::string name)
    : runtime::Tensor(make_shared<descriptor::Tensor>(fromPrecision(precision),
                                                      Shape(shape), "")),
      m_data_pointer(data_pointer),
      m_precision(precision),
      m_layout(layout),
      m_shape(shape),
      m_byte_size(byte_size),
      m_name(name) {}

IETensor::IETensor(std::string name)
    : runtime::Tensor(
          make_shared<descriptor::Tensor>(element::Type(), Shape(), name)) {
  m_data_pointer = nullptr;
  m_byte_size = 0;
  m_name = name;
}

IETensor::~IETensor() {}

void IETensor::write(const void* src, size_t bytes) {
  if (m_data_pointer == nullptr) {
    m_data_pointer = (void*)std::malloc(bytes);
  }
  const int8_t* src_ptr = static_cast<const int8_t*>(src);
  if (src_ptr == nullptr) {
    return;
  }

  std::memcpy((void*)m_data_pointer, src_ptr, bytes);
}

void IETensor::read(void* dst, size_t bytes) const {
  int8_t* dst_ptr = static_cast<int8_t*>(dst);
  if (dst_ptr == nullptr) {
    return;
  }
  if (m_data_pointer == nullptr) {
    return;
  }

  std::memcpy(dst_ptr, m_data_pointer, bytes);
}

const void* IETensor::get_data_ptr() const { return m_data_pointer; }

InferenceEngine::Precision IETensor::get_precision() const {
  return m_precision;
}

InferenceEngine::Layout IETensor::get_layout() const { return m_layout; }

InferenceEngine::SizeVector IETensor::get_dims() const { return m_shape; }

void IETensor::set_name(std::string name) { m_name = name; }

std::string IETensor::get_name() const { return m_name; }

size_t IETensor::get_byte_size() const {
  if (m_data_pointer == nullptr)
    return 0;
  else
    return m_byte_size;
}
}
}
