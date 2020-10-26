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

  m_ie_data =
      std::make_shared<IE_Data>(memory_pointer, precision, layout, shape, size);
}

IETensor::IETensor(const element::Type& element_type, const Shape& shape)
    : IETensor(element_type, shape, nullptr) {}

IETensor::IETensor(const element::Type& element_type, const PartialShape& shape)
    : runtime::Tensor(
          make_shared<descriptor::Tensor>(element_type, shape, "")) {
  throw runtime_error("partial shapes not supported.");
}

IETensor::IETensor(std::shared_ptr<IE_Data> ie_data)
    : runtime::Tensor(make_shared<descriptor::Tensor>(
          fromPrecision(ie_data->get_precision()),
          Shape(ie_data->get_shape()), "")),
      m_ie_data(ie_data) {}

IETensor::~IETensor() {}

void IETensor::write(const void* src, size_t bytes) {
  if (m_ie_data->get_data_ptr() == nullptr) {
    m_ie_data->allocate(bytes);
  }
  m_ie_data->write(src, bytes);
}

void IETensor::read(void* dst, size_t bytes) const {
  m_ie_data->read(dst, bytes);
}

const void* IETensor::get_data_ptr() const {
  return m_ie_data->get_data_ptr();
}
}
}
