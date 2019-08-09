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

// See sample usage in test/test_data_structures.cpp
class IndexLibrary {
 public:
  IndexLibrary(size_t depth);

  void return_index(size_t id);
  int get_index();

  // TODO: if needed implement get_depth() and get_num_free_idxs()
  // Implementing get_depth() might make some sense because if one receives an
  // IndexLibrary object that only gives return_index()==-1 then one might want
  // to know is there any point in waiting for it (it will never return anything
  // other than -1 if depth==0). So the user of the object can query depth and
  // throw an error or take appropriate steps if its 0

 private:
  set<int> m_free_depth_indexes;
  size_t m_depth;
  std::mutex m_mtx;

  void insert_to_free_set(size_t id);
  bool is_free(size_t id);
};

class PipelinedTensorsStore {
 public:
  PipelinedTensorsStore(PipelinedTensorMatrix in, PipelinedTensorMatrix out);

  // returns a tuple of idx, and 2 vectors of ng tensors (input and output
  // groups). If the idx is negative, then its an invalid group (because
  // pipeline is filled right now)
  tuple<int, PipelinedTensorVector, PipelinedTensorVector> get_tensors();

  // Return an integer that was checked out by get_tensors
  void return_tensors(size_t id);

 private:
  PipelinedTensorMatrix m_in_tensors;
  PipelinedTensorMatrix m_out_tensors;
  size_t m_depth;
  size_t m_num_inputs;
  size_t m_num_outputs;
  shared_ptr<IndexLibrary> idx_lib;

  PipelinedTensorVector get_group(bool input, size_t i);
};
}
}