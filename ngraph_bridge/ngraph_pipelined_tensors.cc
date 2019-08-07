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

// TODO change name of file

// Consider an ng executable, which has a inputs and b outputs. Let d_input[i]
// be the depth of the pipeline for input i. Similarly d_output[j] is the depth
// of the pipeline for output j.

// Simplifying assumptions about pipeline depths: for all 0 <= i < a, 0 <= j <
// b, d_input[i] ==  d_output[i] == d. Most likely, d = 2

// Pipelined tensors Matrix: When the executable is used to create tensors, it
// will
// create non-ragged matrices of a x d and b x d tensors.

// Input Group m: A set of a input tensors that can be used to feed data to the
// executable. This represents the m'th column of the input pipelined tensor
// matrix defined above

// Output Group n: A set of b input tensors that can be used to collect data
// from the executable. This represents the n'th column of the input pipelined
// tensor matrix defined above

// Simplifying assumption: We assume m == n, that is we use the same pipeline
// depth index when using call() on an executable. Because of this assumption we
// can store the input and output pipelined tensor matrix in the same class
// object. If we decide we can relax this constraint, then we can split up this
// class into 2, one handling inputs, one for outputs.

#include "ngraph/event_tracing.hpp"
#include "ngraph/runtime/backend.hpp"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

typedef vector<shared_ptr<ng::runtime::Tensor>> PipelinedTensorVector;
typedef vector<PipelinedTensorVector> PipelinedTensorMatrix;

// TODO: unit test the class
class IndexLibrary {
 public:
  IndexLibrary(size_t depth) : m_depth(depth) {
    for (size_t i = 0; i < depth; i++) {
      m_free_depth_indexes.insert(i);
    }
  }

  void return_index(size_t id) {
    if (m_depth == 0) {
      throw std::runtime_error(
          "Depth=0, so no one should be calling return_index");
    } else {
      if (id > m_depth - 1) {
        throw std::runtime_error("Depth = " + to_string(m_depth) +
                                 " but passed an index to return ( = " +
                                 to_string(id) + "), which is too large");
      } else {
        if (is_free(id)) {
          throw std::runtime_error(
              "Attempted to return index " + to_string(id) +
              " but it is already present in the free indices set");
        }
      }
    }
    insert_to_free_set(id);
  }

  size_t get_index() {
    if (m_depth == 0) {
      throw std::runtime_error(
          "Depth=0, so no one should be calling get_index");
    }
    std::lock_guard<std::mutex> lock(m_mtx);
    if (m_free_depth_indexes.size() == 0) {
      return -1;
    } else {
      auto itr = m_free_depth_indexes.begin();
      size_t retval = *itr;
      m_free_depth_indexes.erase(itr);
      return retval;
    }
  }

 private:
  set<int> m_free_depth_indexes;
  size_t m_depth;
  std::mutex m_mtx;

  void insert_to_free_set(size_t id) {
    std::lock_guard<std::mutex> lock(m_mtx);
    m_free_depth_indexes.insert(id);
  }

  bool is_free(size_t id) {
    std::lock_guard<std::mutex> lock(m_mtx);
    if (id > m_depth - 1) {
      throw std::runtime_error("Asked to check if id=" + to_string(id) +
                               " is free, but depth=" + to_string(m_depth));
    }
    return m_free_depth_indexes.find(id) != m_free_depth_indexes.end();
  }
};

class PipelinedTensorsStore {
 public:
  PipelinedTensorsStore(PipelinedTensorMatrix in, PipelinedTensorMatrix out)
      : m_in_tensors(in),
        m_out_tensors(out),
        m_num_inputs(in.size()),
        m_num_outputs(out.size()) {
    if (in.size() > 0) {
      m_depth = in[0].size();
    } else if (out.size() > 0) {
      m_depth = out[0].size();
    } else {
      // The executable has no inputs and outputs
      m_depth = 0;
    }
    idx_lib = make_shared<IndexLibrary>(m_depth);
  }

  // returns a tuple of idx, and 2 vectors of ng tensors (input and output
  // groups). If the idx is negative, then its an invalid group (because
  // pipeline is filled right now)
  tuple<size_t, PipelinedTensorVector, PipelinedTensorVector>
  get_tensors() {
    int i = idx_lib->get_index();
    return make_tuple(i, (i < 0 ? PipelinedTensorVector{}
                                : get_group(true, i)),
                      (i < 0 ? PipelinedTensorVector{}
                             : get_group(false, i)));
  }

  void return_tensors(size_t id) { idx_lib->return_index(id); }

 private:
  PipelinedTensorMatrix m_in_tensors;
  PipelinedTensorMatrix m_out_tensors;
  size_t m_depth;
  size_t m_num_inputs;
  size_t m_num_outputs;

  shared_ptr<IndexLibrary> idx_lib;

  PipelinedTensorVector get_group(bool input, size_t i) {
    PipelinedTensorVector group;
    for (size_t idx = 0; idx < (input ? m_num_inputs : m_num_outputs); idx++) {
      group.push_back((input ? m_in_tensors : m_out_tensors)[idx][i]);
    }
    return group;
  }
};
}
}