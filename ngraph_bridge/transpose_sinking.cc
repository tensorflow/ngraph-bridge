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

#include <algorithm>
#include <iostream>
#include <numeric>
#include <set>
#include <unordered_set>

#include "ngraph/descriptor/input.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/transpose.hpp"
#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"
#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/type.hpp"
#include "ngraph/util.hpp"

#include "logging/ngraph_log.h"
#include "ngraph_bridge/transpose_sinking.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

using TransposeMap =
    unordered_map<shared_ptr<ng::Node>, shared_ptr<ng::opset3::Transpose>>;

void print_transposes_to_delete(
    set<shared_ptr<ng::Node>> transposes_to_delete) {
  NGRAPH_VLOG(0) << "transposes_to_delete";
  for (auto it = transposes_to_delete.begin(); it != transposes_to_delete.end();
       ++it)
    NGRAPH_VLOG(0) << *it;
}

static string describe_node(shared_ptr<ng::Node> node) {
  stringstream ss;
  if (auto transpose = ng::as_type_ptr<ng::opset3::Transpose>(node)) {
    auto t_const = ng::as_type_ptr<ng::opset3::Constant>(
        transpose->input_value(1).get_node_shared_ptr());
    ss << transpose->get_name() << " ( axis order = "
       << ng::vector_to_string(t_const->get_axis_vector_val())
       << " , shape = " << ng::vector_to_string(transpose->get_shape()) << " ) "
       << " , child = " << transpose->get_argument(0)->get_name();
  } else {
    auto reshape = ng::as_type_ptr<ng::opset3::Reshape>(node);
    ss << reshape->get_name()
       << " , shape = " << ng::vector_to_string(reshape->get_shape()) << " ) "
       << " , child = " << reshape->get_argument(0)->get_name();
  }
  return ss.str();
}

static shared_ptr<ng::opset3::Transpose> make_transpose(
    shared_ptr<ng::Node> arg, const ng::AxisVector& input_order) {
  auto ng_input_order = std::make_shared<ng::opset3::Constant>(
      ng::element::u64, ng::Shape{input_order.size()}, input_order);
  auto ng_transpose =
      std::make_shared<ng::opset3::Transpose>(arg, ng_input_order);
  return ng_transpose;
}

static void write_transpose_map(TransposeMap& reorders,
                                shared_ptr<ng::Node> target,
                                shared_ptr<ng::opset3::Transpose> transpose) {
  NGRAPH_VLOG(0) << "Write TransposeMap[" << target->get_name()
                 << "] = " << describe_node(transpose);
  reorders[target] = transpose;
}

static shared_ptr<ng::opset3::Transpose> read_transpose_map(
    TransposeMap& reorders, shared_ptr<ng::Node> target) {
  auto reorder = reorders.at(target);
  NGRAPH_VLOG(0) << "Read TransposeMap[" << target->get_name() << "]  -> "
                 << describe_node(reorder);
  return reorder;
}

static shared_ptr<ng::opset3::Transpose> combine_transposes(
    shared_ptr<ng::opset3::Transpose> t1,
    shared_ptr<ng::opset3::Transpose> t2) {
  auto default_order = ng::get_default_order(t1->get_shape());
  auto t1_const = ng::as_type_ptr<ng::opset3::Constant>(
      t1->input_value(1).get_node_shared_ptr());
  auto t2_const = ng::as_type_ptr<ng::opset3::Constant>(
      t2->input_value(1).get_node_shared_ptr());
  auto perm_t1 =
      ng::apply_permutation(default_order, t1_const->get_axis_vector_val());
  auto perm_t2 =
      ng::apply_permutation(perm_t1, t2_const->get_axis_vector_val());
  auto t_combined = make_transpose(t2->get_argument(0), perm_t2);
  NGRAPH_VLOG(0) << "Combining " << describe_node(t1) << " and "
                 << describe_node(t2) << " into " << describe_node(t_combined);
  return t_combined;
}

static void insert_transpose(shared_ptr<ng::Node> target,
                             shared_ptr<ng::Node> transpose,
                             size_t input_index) {
  NGRAPH_VLOG(0) << "Target " << target->get_name();
  NGRAPH_VLOG(0) << "transpose " << transpose->get_name();
  NGRAPH_VLOG(0) << "target->input(input_index) " << target->input(input_index);
  auto arg = target->input(input_index).get_source_output();
  auto default_order = ng::get_default_order(transpose->get_shape());
  NGRAPH_VLOG(0) << "default_order transpose->get_shape() " << default_order;
  auto const_arg1 = transpose->input(1).get_source_output();
  auto new_transpose = transpose->copy_with_new_inputs({arg, const_arg1});
  NGRAPH_VLOG(0) << "Inserting transpose " << describe_node(new_transpose)
                 << " at input " << target->get_name() << " input index "
                 << input_index;
  target->input(input_index).replace_source_output(new_transpose->output(0));
}

static void delete_transpose(shared_ptr<ng::Node> transpose) {
  NGRAPH_VLOG(0) << "Removing transpose " << transpose->get_name();
  if (!transpose->get_users().empty()) {
    ng::replace_node(transpose, transpose->get_argument(0));
  }
}

static void mark_transpose_for_deletion(
    shared_ptr<ng::Node> transpose,
    set<shared_ptr<ng::Node>>& transposes_to_delete) {
  NGRAPH_VLOG(0) << "Marking transpose " << transpose->get_name()
                 << " for deletion";
  transposes_to_delete.insert(transpose);
  // print_transposes_to_delete(transposes_to_delete);
}

static shared_ptr<ng::opset3::Transpose> create_default_transpose(
    shared_ptr<ng::Node> n) {
  auto default_order = ng::get_default_order(n->get_shape());
  auto default_transpose = make_transpose(n, default_order);
  NGRAPH_VLOG(0) << "Default transpose: " << describe_node(default_transpose);
  return default_transpose;
}

struct Swimmer {
  ng::Input<ng::Node> input;
  shared_ptr<ng::opset3::Transpose> transpose;
};

// Swim is used to push/"swim" reshapes towards paramaters.
// This is typically done for binary ops when
// one operand is in nchw, while  the other one is nhwc
// we prefer nchw since a lot of ngraph ops require this format,
// so keeping things in nchw allows us to eliminate as many reshapes
// as possible
void swim(ng::Input<ng::Node> input,
          shared_ptr<ng::opset3::Transpose> transpose) {
  Swimmer sw{input, transpose};
  list<Swimmer> work_queue;
  work_queue.push_back(sw);

  // TODO: if we support more ops (especially, with >1 args)
  // we will need to keep track of nodes we visited and their reshapes
  while (work_queue.size() > 0) {
    auto csw = work_queue.front();
    work_queue.pop_front();
    auto n_output = csw.input.get_source_output();
    auto n = n_output.get_node_shared_ptr();
    auto materialize = [csw, n_output]() {
      auto n = n_output.get_node_shared_ptr();
      auto new_transpose = csw.transpose->copy_with_new_inputs({n});
      new_transpose->merge_provenance_tags_from(n);
      NGRAPH_VLOG(0) << "Materializing new transpose "
                     << describe_node(new_transpose);
      csw.input.replace_source_output(new_transpose->output(0));
    };  // Only swim past nodes which have a single user
    if (n->get_users().size() > 1) {
      materialize();
      continue;
    }
    NGRAPH_VLOG(0) << "Processing (swimming) " << n->get_name();
    if (n->is_unary_elementwise_arithmetic()) {
      Swimmer nsw{n->input(0), csw.transpose};
      work_queue.push_back(nsw);
      NGRAPH_VLOG(0) << "Propagating reshape " << describe_node(csw.transpose)
                     << " for " << n->get_name() << " to "
                     << n->get_argument(0);
    }
    // TODO: Add cases to push through Reshape and BinaryElementwiseArithmetic
    else {
      // materialize
      materialize();
    }
  }
}

// convert_binary_to_default_order is used when one of the arguments
// of a binary op isn't in the default format (i.e. nhwc instead of nchw)
// We have to normalize this other argument to nchw by swimming nchw towards
// parameters
// as far as we can
static void convert_binary_to_default_order(
    shared_ptr<ng::Node> binary, const ng::Input<ng::Node>& input,
    size_t input_index, shared_ptr<ng::Node> right, TransposeMap& reorders,
    set<shared_ptr<ng::Node>>& tranposes_to_delete) {
  NGRAPH_VLOG(0) << "convert_binary_to_default_order";
  auto left = input.get_source_output().get_node_shared_ptr();
  NGRAPH_VLOG(0) << left;
  NGRAPH_VLOG(0) << "left rank " << (left->get_shape()).size();
  NGRAPH_VLOG(0) << right->get_name();
  NGRAPH_VLOG(0) << "right rank " << (right->get_shape()).size();
  // If the rank for the two inputs is same then we add a Transpose to the
  // parameter
  // otherwise we add a Reshape
  if ((left->get_shape()).size() == (right->get_shape()).size()) {
    NGRAPH_VLOG(0) << reorders.count(right);
    auto r_const = ng::as_type_ptr<ng::opset3::Constant>(
        reorders.at(right)->input_value(1).get_node_shared_ptr());
    auto perm_to_def =
        ng::get_permutation_to_default_order(r_const->get_axis_vector_val());
    NGRAPH_VLOG(0) << "perm_to_def " << perm_to_def;
    // auto new_shape = ng::apply_permutation(left->get_output_shape(0),
    // perm_to_def);
    NGRAPH_VLOG(0) << "right = " << ng::vector_to_string(right->get_shape())
                   << ", " << right->get_name();
    // NGRAPH_VLOG(0) << "new_shape " << new_shape;
    auto new_transpose = make_transpose(left, perm_to_def);
    NGRAPH_VLOG(0) << "left : About to swim " << describe_node(new_transpose)
                   << " up to " << left->get_name();
    // this should now insert and swim transpose on right
    swim(input, new_transpose);
    mark_transpose_for_deletion(reorders.at(right), tranposes_to_delete);
    write_transpose_map(reorders, binary, read_transpose_map(reorders, right));
  } else {
    NGRAPH_VLOG(0) << "Inside ELSE - CREATING RESHAPE INSTEAD";
    std::vector<size_t> reshape_pattern_values((left->get_shape()).size(), 1U);
    reshape_pattern_values[1] = (right->get_shape()).front();
    auto reshape_pattern = make_shared<ng::opset3::Constant>(
        ng::element::u64, ng::Shape{reshape_pattern_values.size()},
        reshape_pattern_values);
    auto ng_reshaped =
        std::make_shared<ng::opset3::Reshape>(right, reshape_pattern, false);

    NGRAPH_VLOG(0) << "Inserting reshape at input " << binary->get_name()
                   << " at index " << input_index;
    auto arg = binary->input(input_index).get_source_output();
    auto const_arg1 = ng_reshaped->input(1).get_source_output();
    NGRAPH_VLOG(0) << arg << " Arg shape: " << arg.get_shape();
    auto new_reshape = ng_reshaped->copy_with_new_inputs({arg, const_arg1});
    NGRAPH_VLOG(0) << "Inserting reshape " << describe_node(ng_reshaped)
                   << " at input " << binary->get_name();
    binary->input(input_index).replace_source_output(new_reshape->output(0));

    mark_transpose_for_deletion(read_transpose_map(reorders, right),
                                tranposes_to_delete);
    mark_transpose_for_deletion(reorders.at(left), tranposes_to_delete);
    write_transpose_map(reorders, binary, reorders.at(left));
  }
}

static void materialize_shapes(
    shared_ptr<ng::Node> n, TransposeMap& reorders,
    set<shared_ptr<ng::Node>>& transposes_to_delete) {
  NGRAPH_VLOG(0) << "Materialize shapes";
  // skip multiple output nodes and deal with GOEs exclusively
  if (n->get_output_size() > 1) {
    return;
  }

  for (size_t i = 0; i < n->get_input_size(); i++) {
    // materialize all pending reshapes, flush pending reshapes
    auto arg = n->input_value(i).get_node_shared_ptr();
    NGRAPH_VLOG(0) << "arg->get_name(): " << arg->get_name();
    if (reorders.count(arg) != 0) {
      NGRAPH_VLOG(0) << "reorders.count(arg) != 0 " << reorders.count(arg);  // 1
      auto arg_transpose = reorders.at(arg);
      NGRAPH_VLOG(0) << "arg_transpose->get_name(): "
                     << arg_transpose->get_name();  // Transpose_251
      NGRAPH_VLOG(0) << "Materializing " << describe_node(arg_transpose)
                     << " for " << arg->get_name();
      mark_transpose_for_deletion(arg_transpose, transposes_to_delete);
      auto arg_transpose_shape = arg_transpose->get_shape();
      auto t_const = ng::as_type_ptr<ng::opset3::Constant>(
          arg_transpose->input_value(1).get_node_shared_ptr());
      NGRAPH_VLOG(0) << "t_const->get_axis_vector_val() "
                     << t_const->get_axis_vector_val();
      NGRAPH_VLOG(0) << "ng::get_default_order(arg->get_output_shape(0) "
                     << ng::get_default_order(arg->get_output_shape(0));
      // Since
      if (ng::get_default_order(arg->get_output_shape(0)) !=
          t_const->get_axis_vector_val()) {
        NGRAPH_VLOG(0) << "arg_OUTPUT_shape: " << arg->get_output_shape(0)
                       << " arg_transpose_shape " << arg_transpose_shape;
        // Insert if arg needs to be transposed.
        insert_transpose(n, arg_transpose, i);
      }
      // no swimming up
    }
  }
  write_transpose_map(reorders, n, create_default_transpose(n));
}

static void sink_transpose(shared_ptr<ng::opset3::Transpose> transpose,
                           TransposeMap& reorders,
                           set<shared_ptr<ng::Node>>& transposes_to_delete) {
  NGRAPH_VLOG(0) << "Sinking transpose :" << describe_node(transpose);
  auto orig_transpose = reorders.at(transpose->get_argument(0));
  NGRAPH_VLOG(0) << "Sinking transpose :" << describe_node(orig_transpose);
  // 1) Rank changing operation ? but we would not have any rank changing
  // transposes
  // so this code should never be excecuted
  // Why compare output rank with #of inputs?
  // if (transpose->get_output_shape(0).size() != transpose->get_input_size())
  // {
  //     NGRAPH_VLOG(0) << "Materializing " << describe_node(orig_transpose) <<
  //     " for reshape "
  //                  << describe_node(transpose);
  //     insert_transpose(transpose, orig_transpose, 0);
  //     mark_transpose_for_deletion(orig_transpose, transposes_to_delete);
  //     write_transpose_map(reorders, transpose,
  //     create_default_transpose(transpose));
  // }
  // else
  // {
  // combine both reshapes
  auto new_transpose = combine_transposes(orig_transpose, transpose);
  // remove original reshape now it's combined with a new one
  // should be safe to remove an already detached node
  mark_transpose_for_deletion(orig_transpose, transposes_to_delete);
  // replace reshape with combined one
  ng::replace_node(transpose, new_transpose);
  mark_transpose_for_deletion(transpose, transposes_to_delete);
  write_transpose_map(reorders, new_transpose, new_transpose);
  // }
}

static void sink_unary(shared_ptr<ng::Node> n, TransposeMap& reorders,
                       set<shared_ptr<ng::Node>>& /* reshapes_to_delete */) {
  NGRAPH_VLOG(0) << "sink_unary";
  auto arg_reshape = read_transpose_map(reorders, n->get_argument(0));
  NGRAPH_VLOG(0) << "Propagating " << describe_node(arg_reshape) << " for "
                 << n->get_name();
  write_transpose_map(reorders, n, arg_reshape);
}

static void sink_binary(shared_ptr<ng::Node> binary, TransposeMap& reorders,
                        set<shared_ptr<ng::Node>>& transposes_to_delete) {
  NGRAPH_VLOG(0) << "sink_binary";
  auto left = binary->get_argument(0);   // Transpose262
  auto right = binary->get_argument(1);  // Param232
  auto left_shape = reorders.at(left)->get_shape();
  auto right_shape = reorders.at(right)->get_shape();
  NGRAPH_VLOG(0) << "LEFT " << ng::get_default_order(left_shape);
  NGRAPH_VLOG(0) << "RIGHT " << ng::get_default_order(right_shape);
  auto l_const = ng::as_type_ptr<ng::opset3::Constant>(
      reorders.at(left)->input_value(1).get_node_shared_ptr());
  auto r_const = ng::as_type_ptr<ng::opset3::Constant>(
      reorders.at(right)->input_value(1).get_node_shared_ptr());
  NGRAPH_VLOG(0) << "L axis order " << l_const->get_axis_vector_val();
  NGRAPH_VLOG(0) << "R axis order " << r_const->get_axis_vector_val();

  // Check if the reorders have same rank because if not then one of the inputs
  // needs a Reshape and not a Transpose
  if (left->get_shape().size() != right->get_shape().size()) {
    // figure out where to add the reshape
    if (left->get_shape().size() != 4) {
      NGRAPH_VLOG(0) << "Adding Reshape to left";
      // reshape needs to be added for the left node
      convert_binary_to_default_order(binary, binary->input(1), 0, left,
                                      reorders, transposes_to_delete);
    } else {
      NGRAPH_VLOG(0) << "Adding Reshape to right";
      // reshape needs to be added for the right node
      convert_binary_to_default_order(binary, binary->input(0), 1, right,
                                      reorders, transposes_to_delete);
    }
  } else {
    if (l_const->get_axis_vector_val() == r_const->get_axis_vector_val()) {
      NGRAPH_VLOG(0) << "Propagating " << describe_node(reorders.at(left))
                     << " for " << binary->get_name();
      //     write_transpose_map(reorders, binary, read_transpose_map(reorders,
      //     left));
      //     // at this point, both reshapes will be eventually removed
      //     mark_transpose_for_deletion(reorders.at(left), reshapes_to_delete);
      //     mark_transpose_for_deletion(reorders.at(right),
      //     reshapes_to_delete);
    } else if (l_const->get_axis_vector_val() ==
               ng::get_default_order(left->get_shape())) {
      NGRAPH_VLOG(0) << "Frist else";
      // convert_binary_to_default_order(
      //     binary, binary->input(0), 1, right, reorders,
      //     transposes_to_delete);
    } else if (r_const->get_axis_vector_val() ==
               ng::get_default_order(right->get_shape())) {
      NGRAPH_VLOG(0) << "secomd else";
      NGRAPH_VLOG(0) << binary->input(1);
      NGRAPH_VLOG(0) << binary->input(0);
      convert_binary_to_default_order(binary, binary->input(1), 0, left,
                                      reorders, transposes_to_delete);
    } else {
      NGRAPH_VLOG(0) << "Third else";
      //     NGRAPH_DEBUG << "Materializing both reshapes for " <<
      //     binary->get_name();
      //     NGRAPH_DEBUG << "Left = " << describe_node(reorders.at(left));
      //     NGRAPH_DEBUG << "Right = " << describe_node(reorders.at(right));
      //     mark_transpose_for_deletion(reorders.at(left),
      //     transposes_to_delete);
      //     mark_transpose_for_deletion(reorders.at(right),
      //     transposes_to_delete);
      //     insert_transpose(binary, reorders.at(left), 0);
      //     insert_transpose(binary, reorders.at(right), 1);
    }
  }
}

static void sink_pad(shared_ptr<ng::op::Pad> n, TransposeMap& reorders,
                     set<shared_ptr<ng::Node>>& /* reshapes_to_delete */) {
  // auto arg_reshape = reorders.at(n->get_argument(0));
  // auto order = arg_reshape->get_input_order();
  // // we need the correct input shape to produce the right output shape
  // // we are going to create a label of the right input shape,
  // // so a new pad will have the right shape
  // auto def_order = ng::get_permutation_to_default_order(order);
  // auto input_shape = ng::apply_permutation(arg_reshape->get_shape(),
  // def_order);
  // auto dummy_correct_shape =
  //     make_shared<ng::pattern::op::Label>(arg_reshape->get_element_type(),
  //     input_shape);

  // auto new_lower = ng::apply_permutation(n->get_padding_below(), def_order);
  // auto new_upper = ng::apply_permutation(n->get_padding_above(), def_order);
  // auto new_pad = make_shared<ng::op::Pad>(
  //     dummy_correct_shape, n->get_argument(1), new_lower, new_upper,
  //     n->get_pad_mode());
  // ng::replace_node(dummy_correct_shape, n->get_argument(0));
  // NGRAPH_DEBUG << "Replacing " << n->get_name() << " with " <<
  // new_pad->get_name();
  // ng::replace_node(n, new_pad);
  // auto new_reshape = make_transpose(new_pad, order, n->get_shape());
  // NGRAPH_DEBUG << "Propagating " << describe_node(new_reshape) << " for " <<
  // n->get_name();
  // write_transpose_map(reorders, new_pad, new_reshape);
}

// The goal of ReshapeSinking is to remove
// round-trip reshapes(i.e. nhwc->nchw(nchw-only-op)->nhwc)
// around nchw-only-op (e.g.Convolution, Batchnorm, Avg/MaxPool)
// This is achieved by both **sinking**, propagating reshapes
// through ops towards op::Results,
// or **swimming** Reshapes up towards op::Parameter
// For each op type we support we can either combine
// two reshapes by replacing the existing Reshape,
// materialize pending reshapes if they can't be propagated through op
bool TransposeSinking::run_on_function(shared_ptr<ng::Function> f) {
  TransposeMap reorders;
  ng::NodeVector results;
  set<shared_ptr<ng::Node>> transposes_to_delete;

  // STEP 1 : Sink or Swim reshapes away for op clusters
  for (auto n : f->get_ordered_ops()) {
    NGRAPH_VLOG(0) << "Start: Processing node " << n->get_name();
    // collect all Result nodes for a sanity check
    if (n->is_output()) {
      results.push_back(n);
    }

    if (auto transpose = ng::as_type_ptr<ng::opset3::Transpose>(n)) {
      sink_transpose(transpose, reorders, transposes_to_delete);
    } else if (n->is_unary_elementwise_arithmetic()) {
      sink_unary(n, reorders, transposes_to_delete);
    } else if (n->is_binary_elementwise_arithmetic()) {
      sink_binary(n, reorders, transposes_to_delete);
    }
    // else if (auto pad = ng::as_type_ptr<ng::op::Pad>(n))
    // {
    //     sink_pad(pad, reorders, reshapes_to_delete);
    // }
    else {
      materialize_shapes(n, reorders, transposes_to_delete);
    }
    NGRAPH_VLOG(0) << "End: Processing node " << n->get_name();
  }

  // STEP 2: purge all the reshapes we either sunk or swam.
  for (auto r : transposes_to_delete) {
    delete_transpose(r);
  }

  NGRAPH_VLOG(0) << "END: STEP 2 ";
  // make sure shapes are always materialized before results
  for (auto r : results) {
    NGRAPH_VLOG(0) << r;
    NGRAPH_CHECK(
        r->get_shape() == r->get_input_shape(0) &&
            r->get_element_type() == r->get_argument(0)->get_element_type(),
        " op::Result = ", *r, ", Arg = ", *r->get_argument(0));
  }

  // STEP 3: fix wrong shape info wholesale
  for (auto n : f->get_ordered_ops()) {
    n->revalidate_and_infer_types();
  }
  return true;
}

}  // namespace ngraph_bridge

}  // namespace tensorflow