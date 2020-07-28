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

#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/util.hpp"

#include "logging/ngraph_log.h"
#include "ngraph_bridge/pass/transpose_sinking.h"

using namespace std;

namespace tensorflow {
namespace ngraph_bridge {

using TransposeMap =
    unordered_map<shared_ptr<ngraph::Node>, shared_ptr<ngraph::op::Reshape>>;

static ngraph::CoordinateDiff apply_permutation(ngraph::CoordinateDiff input,
                                                ngraph::AxisVector order) {
  ngraph::CoordinateDiff output(input.size());
  for (size_t i = 0; i < order.size(); i++) {
    output[i] = input.at(order.at(i));
  }
  return output;
}

static string describe_transpose(shared_ptr<ngraph::Node> node) {
  stringstream ss;
  auto reshape = ngraph::as_type_ptr<ngraph::op::Reshape>(node);
  ss << reshape->get_name() << " ( axis order = "
     << ngraph::vector_to_string(reshape->get_input_order())
     << " , shape = " << ngraph::vector_to_string(reshape->get_shape()) << " ) "
     << " , child = " << reshape->get_argument(0)->get_name();

  return ss.str();
}

static shared_ptr<ngraph::op::Reshape> make_transpose(
    shared_ptr<ngraph::Node> arg, const ngraph::AxisVector& input_order,
    const ngraph::Shape& output_shape) {
  auto reshape =
      make_shared<ngraph::op::Reshape>(arg, input_order, output_shape);
  NGRAPH_VLOG(4) << "Make Reshape " << describe_transpose(reshape);
  return reshape;
}

static void write_transposemap(TransposeMap& reorders,
                               shared_ptr<ngraph::Node> target,
                               shared_ptr<ngraph::op::Reshape> reshape) {
  NGRAPH_VLOG(4) << "Write TransposeMap[" << target->get_name()
                 << "] = " << describe_transpose(reshape);
  reorders[target] = reshape;
}

static shared_ptr<ngraph::op::Reshape> read_transposemap(
    TransposeMap& reorders, shared_ptr<ngraph::Node> target) {
  auto reorder = reorders.at(target);
  NGRAPH_VLOG(4) << "Read TransposeMap[" << target->get_name() << "]  -> "
                 << describe_transpose(reorder);
  return reorder;
}

static shared_ptr<ngraph::op::Reshape> combine_transposes(
    shared_ptr<ngraph::op::Reshape> r1, shared_ptr<ngraph::op::Reshape> r2) {
  auto default_order = ngraph::get_default_order(r1->get_shape());
  auto perm_r1 =
      ngraph::apply_permutation(default_order, r1->get_input_order());
  auto perm_r2 = ngraph::apply_permutation(perm_r1, r2->get_input_order());
  auto rreshape = make_transpose(r2->get_argument(0), perm_r2, r2->get_shape());
  NGRAPH_VLOG(4) << "Combining " << describe_transpose(r1) << " and "
                 << describe_transpose(r2) << " into "
                 << describe_transpose(rreshape);
  return rreshape;
}

static void insert_transpose(shared_ptr<ngraph::Node> target,
                             shared_ptr<ngraph::Node> reshape,
                             size_t input_index) {
  NGRAPH_VLOG(4) << "Inserting reshape at input " << target->get_name()
                 << " input index " << input_index;
  auto arg = target->input(input_index).get_source_output();
  NGRAPH_VLOG(4) << "Arg shape: " << arg.get_shape();
  auto new_reshape = reshape->copy_with_new_inputs({arg});
  NGRAPH_VLOG(4) << "Inserting reshape " << describe_transpose(new_reshape)
                 << " at input " << target->get_name() << " input index "
                 << input_index;
  target->input(input_index).replace_source_output(new_reshape->output(0));
}

static void delete_reshape(shared_ptr<ngraph::Node> reshape) {
  NGRAPH_VLOG(4) << "Removing reshape " << reshape->get_name();
  if (!reshape->get_users().empty()) {
    ngraph::replace_node(reshape, reshape->get_argument(0));
  }
}

static void mark_transpose_for_deletion(
    shared_ptr<ngraph::Node> reshape,
    set<shared_ptr<ngraph::Node>>& reshapes_to_delete) {
  NGRAPH_VLOG(4) << "Marking reshape " << reshape->get_name()
                 << " for deletion";
  reshapes_to_delete.insert(reshape);
}

static shared_ptr<ngraph::op::Reshape> create_default_transpose(
    shared_ptr<ngraph::Node> n) {
  auto default_order = ngraph::get_default_order(n->get_shape());
  auto default_reshape = make_transpose(n, default_order, n->get_shape());
  NGRAPH_VLOG(4) << "Default reshape: " << describe_transpose(default_reshape);
  return default_reshape;
}

// compute an axis order that converts the given axis order to default
static ngraph::AxisSet get_quantization_axes_in_default_order(
    shared_ptr<ngraph::op::Reshape> arg_reshape,
    const ngraph::AxisSet& old_axis_set) {
  auto perm_to_def =
      ngraph::get_permutation_to_default_order(arg_reshape->get_input_order());
  ngraph::AxisSet axis_set;
  for (auto axis : old_axis_set) {
    axis_set.insert(perm_to_def.at(axis));
  }
  return axis_set;
}

struct Swimmer {
  ngraph::Input<ngraph::Node> input;
  shared_ptr<ngraph::op::Reshape> reshape;
};

// Swim is used to push/"swim" reshapes towards paramaters.
// This is typically done for binary ops when
// one operand is in nchw, while  the other one is nhwc
// we prefer nchw since a lot of ngraph ops require this format,
// so keeping things in nchw allows us to eliminate as many reshapes
// as possible
void swim(ngraph::Input<ngraph::Node> input,
          shared_ptr<ngraph::op::Reshape> reshape) {
  Swimmer sw{input, reshape};
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
      auto new_reshape = csw.reshape->clone_with_new_inputs({n});
      new_reshape->merge_provenance_tags_from(n);
      NGRAPH_VLOG(4) << "Materializing new reshape "
                     << describe_transpose(new_reshape);
      csw.input.replace_source_output(new_reshape->output(0));
    };  // Only swim past nodes which have a single user
    if (n->get_users().size() > 1) {
      materialize();
      continue;
    }
    NGRAPH_VLOG(4) << "Processing (swimming) " << n->get_name();
    if (n->is_unary_elementwise_arithmetic()) {
      Swimmer nsw{n->input(0), csw.reshape};
      work_queue.push_back(nsw);
      NGRAPH_VLOG(4) << "Propagating reshape "
                     << describe_transpose(csw.reshape) << " for "
                     << n->get_name() << " to " << n->get_argument(0);
    } else if (ngraph::is_type<ngraph::op::Broadcast>(n)) {
      auto old_broadcast = static_pointer_cast<ngraph::op::Broadcast>(n);
      auto broadcast_axes = old_broadcast->get_broadcast_axes();
      auto broadcast_reshape = csw.reshape;
      // swimming can only handle 1 dim change
      if (broadcast_reshape->get_shape().size() -
              old_broadcast->get_shape().size() >
          1) {
        materialize();
        continue;
      }
      bool in_order = true;
      ngraph::AxisSet new_broadcast_axes;
      vector<size_t> new_source_axes;
      auto input_order = broadcast_reshape->get_input_order();
      for (size_t i = 0; i < input_order.size(); i++) {
        if (broadcast_axes.count(input_order.at(i)) != 0) {
          new_broadcast_axes.insert(i);
        } else {
          if (new_source_axes.size() != 0 &&
              new_source_axes.back() > input_order.at(i)) {
            in_order = false;
          }
          new_source_axes.push_back(i);
        }
      }

      auto broadcast_input = old_broadcast->get_argument(0);
      if (!in_order) {
        ngraph::AxisVector new_source_axes_sorted{new_source_axes};
        sort(new_source_axes_sorted.begin(), new_source_axes_sorted.end());
        map<size_t, size_t> old_new_source_axes;
        for (size_t i = 0; new_source_axes_sorted.size(); i++) {
          old_new_source_axes.insert({new_source_axes.at(i), i});
        }

        ngraph::AxisVector new_source_axis_order;
        for (auto axis : new_source_axes_sorted) {
          new_source_axis_order.push_back(old_new_source_axes.at(axis));
        }

        auto new_arg_shape = ngraph::apply_permutation(
            broadcast_input->get_shape(), new_source_axis_order);
        broadcast_input = make_transpose(broadcast_input, new_source_axis_order,
                                         new_arg_shape);
      }

      auto new_broadcast = make_shared<ngraph::op::Broadcast>(
          broadcast_input, broadcast_reshape->get_shape(), new_broadcast_axes);
      csw.input.replace_source_output(new_broadcast->output(0));
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
    shared_ptr<ngraph::Node> binary, const ngraph::Input<ngraph::Node>& input,
    shared_ptr<ngraph::Node> right, TransposeMap& reorders,
    set<shared_ptr<ngraph::Node>>& reshapes_to_delete) {
  auto left = input.get_source_output().get_node_shared_ptr();
  auto perm_to_def = ngraph::get_permutation_to_default_order(
      reorders.at(right)->get_input_order());
  auto new_shape = ngraph::apply_permutation(left->get_shape(), perm_to_def);
  NGRAPH_VLOG(4) << "right = " << ngraph::vector_to_string(right->get_shape())
                 << ", " << right->get_name();
  auto new_reshape = make_transpose(left, perm_to_def, new_shape);
  NGRAPH_VLOG(4) << "left : About to swim " << describe_transpose(new_reshape)
                 << " up to " << left->get_name();
  // this should now insert and swim reshape on right
  swim(input, new_reshape);
  mark_transpose_for_deletion(reorders.at(right), reshapes_to_delete);
  write_transposemap(reorders, binary, read_transposemap(reorders, right));
}

static void materialize_shapes(
    shared_ptr<ngraph::Node> n, TransposeMap& reorders,
    set<shared_ptr<ngraph::Node>>& reshapes_to_delete) {
  // skip multiple output nodes and deal with GOEs exclusively
  if (n->get_output_size() > 1) {
    return;
  }

  for (size_t i = 0; i < n->get_arguments().size(); i++) {
    // materialize all pending reshapes, flush pending reshapes
    auto arg = n->get_argument(i);
    if (reorders.count(arg) != 0) {
      auto arg_reshape = reorders.at(arg);
      NGRAPH_VLOG(4) << "Materializing " << describe_transpose(arg_reshape)
                     << " for " << arg->get_name();
      mark_transpose_for_deletion(arg_reshape, reshapes_to_delete);
      auto arg_shape = arg->get_shape();
      if (arg_reshape->get_input_order() !=
          get_default_order(arg->get_shape())) {
        // Insert if arg needs to be transposed.
        insert_transpose(n, arg_reshape, i);
      }
      // no swimming up
    }
  }
  write_transposemap(reorders, n, create_default_transpose(n));
}

static void sink_transpose(shared_ptr<ngraph::op::Reshape> reshape,
                         TransposeMap& reorders,
                         set<shared_ptr<ngraph::Node>>& reshapes_to_delete) {
  NGRAPH_VLOG(4) << "Sinking Reshape :" << describe_transpose(reshape);
  auto orig_reshape = reorders.at(reshape->get_argument(0));
  // 1) Not a Transpose or 2) Rank changing operation.
  if ((reshape->get_output_shape(0).size() !=
       reshape->get_input_order().size()) ||
      (!reshape->get_is_transpose())) {
    NGRAPH_VLOG(4) << "Materializing " << describe_transpose(orig_reshape)
                   << " for reshape " << describe_transpose(reshape);
    insert_transpose(reshape, orig_reshape, 0);
    mark_transpose_for_deletion(orig_reshape, reshapes_to_delete);
    write_transposemap(reorders, reshape, create_default_transpose(reshape));
  } else {
    // combine both reshapes
    auto new_reshape = combine_transposes(orig_reshape, reshape);
    // remove original reshape now it's combined with a new one
    // should be safe to remove an already detached node
    mark_transpose_for_deletion(orig_reshape, reshapes_to_delete);
    // replace reshape with combined one
    ngraph::replace_node(reshape, new_reshape);
    mark_transpose_for_deletion(new_reshape, reshapes_to_delete);
    write_transposemap(reorders, new_reshape, new_reshape);
  }
}

static void sink_unary(
    shared_ptr<ngraph::Node> n, TransposeMap& reorders,
    set<shared_ptr<ngraph::Node>>& /* reshapes_to_delete */) {
  auto arg_reshape = read_transposemap(reorders, n->get_argument(0));
  NGRAPH_VLOG(4) << "Propagating " << describe_transpose(arg_reshape) << " for "
                 << n->get_name();
  write_transposemap(reorders, n, arg_reshape);
}

static void sink_binary(shared_ptr<ngraph::Node> binary, TransposeMap& reorders,
                        set<shared_ptr<ngraph::Node>>& reshapes_to_delete) {
  auto left = binary->get_argument(0);
  auto right = binary->get_argument(1);

  if (reorders.at(left)->get_input_order() ==
      reorders.at(right)->get_input_order()) {
    NGRAPH_VLOG(4) << "Propagating " << describe_transpose(reorders.at(left))
                   << " for " << binary->get_name();
    write_transposemap(reorders, binary, read_transposemap(reorders, left));
    // at this point, both reshapes will be eventually removed
    mark_transpose_for_deletion(reorders.at(left), reshapes_to_delete);
    mark_transpose_for_deletion(reorders.at(right), reshapes_to_delete);
  } else if (reorders.at(left)->get_input_order() ==
             ngraph::get_default_order(left->get_shape())) {
    convert_binary_to_default_order(binary, binary->input(0), right, reorders,
                                    reshapes_to_delete);
  } else if (reorders.at(right)->get_input_order() ==
             ngraph::get_default_order(right->get_shape())) {
    convert_binary_to_default_order(binary, binary->input(1), left, reorders,
                                    reshapes_to_delete);
  } else {
    NGRAPH_VLOG(4) << "Materializing both reshapes for " << binary->get_name();
    NGRAPH_VLOG(4) << "Left = " << describe_transpose(reorders.at(left));
    NGRAPH_VLOG(4) << "Right = " << describe_transpose(reorders.at(right));
    mark_transpose_for_deletion(reorders.at(left), reshapes_to_delete);
    mark_transpose_for_deletion(reorders.at(right), reshapes_to_delete);
    insert_transpose(binary, reorders.at(left), 0);
    insert_transpose(binary, reorders.at(right), 1);
  }
}

static void sink_slice(
    shared_ptr<ngraph::op::Slice> n, TransposeMap& reorders,
    set<shared_ptr<ngraph::Node>>& /* reshapes_to_delete */) {
  auto arg_reshape = reorders.at(n->get_argument(0));
  auto order = arg_reshape->get_input_order();

  // we need the correct input shape to produce the right output shape
  // we are going to create a label of the right input shape,
  // so a new slice will have the right shape
  auto def_order = ngraph::get_permutation_to_default_order(order);
  auto input_shape =
      ngraph::apply_permutation(arg_reshape->get_shape(), def_order);
  auto dummy_correct_shape = make_shared<ngraph::pattern::op::Label>(
      arg_reshape->get_element_type(), input_shape);

  auto new_lower = ngraph::apply_permutation(n->get_lower_bounds(), def_order);
  auto new_upper = ngraph::apply_permutation(n->get_upper_bounds(), def_order);
  auto new_strides = ngraph::apply_permutation(n->get_strides(), def_order);
  auto new_slice = make_shared<ngraph::op::Slice>(
      dummy_correct_shape, new_lower, new_upper, new_strides);
  ngraph::replace_node(dummy_correct_shape, n->get_argument(0));
  NGRAPH_VLOG(4) << "Replacing " << n->get_name() << " with "
                 << new_slice->get_name();
  ngraph::replace_node(n, new_slice);

  auto new_reshape = make_transpose(new_slice, order, n->get_shape());
  NGRAPH_VLOG(4) << "Propagating " << describe_transpose(new_reshape) << " for "
                 << n->get_name();
  write_transposemap(reorders, new_slice, new_reshape);
}

static void sink_pad(shared_ptr<ngraph::op::Pad> n, TransposeMap& reorders,
                     set<shared_ptr<ngraph::Node>>& /* reshapes_to_delete */) {
  auto arg_reshape = reorders.at(n->get_argument(0));
  auto order = arg_reshape->get_input_order();
  // we need the correct input shape to produce the right output shape
  // we are going to create a label of the right input shape,
  // so a new pad will have the right shape
  auto def_order = ngraph::get_permutation_to_default_order(order);
  auto input_shape =
      ngraph::apply_permutation(arg_reshape->get_shape(), def_order);
  auto dummy_correct_shape = make_shared<ngraph::pattern::op::Label>(
      arg_reshape->get_element_type(), input_shape);
  auto new_lower = apply_permutation(n->get_padding_below(), def_order);
  auto new_upper = apply_permutation(n->get_padding_above(), def_order);
  auto new_pad =
      make_shared<ngraph::op::Pad>(dummy_correct_shape, n->get_argument(1),
                                   new_lower, new_upper, n->get_pad_mode());
  ngraph::replace_node(dummy_correct_shape, n->get_argument(0));
  NGRAPH_VLOG(4) << "Replacing " << n->get_name() << " with "
                 << new_pad->get_name();
  ngraph::replace_node(n, new_pad);
  auto new_reshape = make_transpose(new_pad, order, n->get_shape());
  NGRAPH_VLOG(4) << "Propagating " << describe_transpose(new_reshape) << " for "
                 << n->get_name();
  write_transposemap(reorders, new_pad, new_reshape);
}
static void sink_quantize(
    shared_ptr<ngraph::op::Quantize> quantize, TransposeMap& reorders,
    set<shared_ptr<ngraph::Node>>& /* reshapes_to_delete */) {
  auto arg_reshape = reorders.at(quantize->get_argument(0));
  ngraph::AxisSet axes_in_def_order =
      get_quantization_axes_in_default_order(arg_reshape, quantize->get_axes());
  auto new_quantize = make_shared<ngraph::op::Quantize>(
      quantize->get_argument(0), quantize->get_argument(1),
      quantize->get_argument(2), quantize->get_element_type(),
      axes_in_def_order, quantize->get_round_mode());

  ngraph::replace_node(quantize, new_quantize);
  write_transposemap(reorders, new_quantize, arg_reshape);
}

static void sink_concat(shared_ptr<ngraph::op::Concat> n,
                        TransposeMap& reorders,
                        set<shared_ptr<ngraph::Node>>& reshapes_to_delete) {
  auto arg_reshape = reorders.at(n->get_argument(0));
  auto order = arg_reshape->get_input_order();
  // we need the correct input shape to produce the right output shape
  // we are going to create a label of the right input shape,
  // so a new slice will have the right shape
  auto def_order = ngraph::get_permutation_to_default_order(order);
  auto input_shape =
      ngraph::apply_permutation(arg_reshape->get_shape(), def_order);
  auto dummy_correct_shape = make_shared<ngraph::pattern::op::Label>(
      arg_reshape->get_element_type(), input_shape);

  ngraph::NodeVector new_args;
  new_args.push_back(dummy_correct_shape);

  for (size_t i = 1; i < n->get_input_size(); i++) {
    auto iarg_reshape = reorders.at(n->get_argument(i));
    auto iorder = iarg_reshape->get_input_order();
    if (iorder != order) {
      NGRAPH_VLOG(4) << " input order at " << i
                     << "-th arg is different from first arg";
      materialize_shapes(n, reorders, reshapes_to_delete);
      return;
    }

    auto iinput_shape =
        ngraph::apply_permutation(iarg_reshape->get_shape(), def_order);
    auto idummy_correct_shape = make_shared<ngraph::pattern::op::Label>(
        iarg_reshape->get_element_type(), iinput_shape);
    new_args.push_back(idummy_correct_shape);
  }

  auto new_axis = order.at(n->get_concatenation_axis());
  auto new_concat = make_shared<ngraph::op::Concat>(new_args, new_axis);
  // put back the original arguments
  for (size_t i = 0; i < new_concat->get_input_size(); i++) {
    ngraph::replace_node(new_args.at(i), n->get_argument(i));
  }
  NGRAPH_VLOG(4) << "Replacing " << n->get_name() << " with "
                 << new_concat->get_name();
  ngraph::replace_node(n, new_concat);

  auto new_reshape = make_transpose(new_concat, order, n->get_shape());
  NGRAPH_VLOG(4) << "Propagating " << describe_transpose(new_reshape) << " for "
                 << n->get_name();
  write_transposemap(reorders, new_concat, new_reshape);
}

static void sink_dequantize(
    shared_ptr<ngraph::op::Dequantize> dequantize, TransposeMap& reorders,
    set<shared_ptr<ngraph::Node>>& /* reshapes_to_delete */) {
  auto arg_reshape = reorders.at(dequantize->get_argument(0));
  ngraph::AxisSet axes_in_def_order = get_quantization_axes_in_default_order(
      arg_reshape, dequantize->get_axes());
  auto new_dequantize = make_shared<ngraph::op::Dequantize>(
      dequantize->get_argument(0), dequantize->get_argument(1),
      dequantize->get_argument(2), dequantize->get_element_type(),
      axes_in_def_order);

  ngraph::replace_node(dequantize, new_dequantize);
  write_transposemap(reorders, new_dequantize, arg_reshape);
}

// The goal of ReshapeSinking is to remove
// round-trip reshapes(i.e. nhwc->nchw(nchw-only-op)->nhwc)
// around nchw-only-op (e.g.Convolution, Batchnorm, Avg/MaxPool)
// This is achieved by both **sinking**, propagating reshapes
// through ops towards ngraph::op::Results,
// or **swimming** Reshapes up towards ngraph::op::Parameter
// For each op type we support we can either combine
// two reshapes by replacing the existing Reshape,
// materialize pending reshapes if they can't be propagated through op
bool TransposeSinking::run_on_function(shared_ptr<ngraph::Function> f) {
  TransposeMap reorders;
  ngraph::NodeVector results;
  set<shared_ptr<ngraph::Node>> reshapes_to_delete;

  // STEP 1 : Sink or Swim reshapes away for op clusters
  for (auto n : f->get_ordered_ops()) {
    NGRAPH_VLOG(4) << "Start: Processing node " << n->get_name();
    // collect all Result nodes for a sanity check
    if (n->is_output()) {
      results.push_back(n);
    }

    if (auto reshape = ngraph::as_type_ptr<ngraph::op::Reshape>(n)) {
      sink_transpose(reshape, reorders, reshapes_to_delete);
    } else if (n->is_unary_elementwise_arithmetic()) {
      sink_unary(n, reorders, reshapes_to_delete);
    } else if (n->is_binary_elementwise_arithmetic()) {
      sink_binary(n, reorders, reshapes_to_delete);
    } else if (auto goe =
                   ngraph::as_type_ptr<ngraph::op::GetOutputElement>(n)) {
      write_transposemap(reorders, goe, create_default_transpose(goe));
    } else if (auto quantize = ngraph::as_type_ptr<ngraph::op::Quantize>(n)) {
      sink_quantize(quantize, reorders, reshapes_to_delete);
    } else if (auto dequantize =
                   ngraph::as_type_ptr<ngraph::op::Dequantize>(n)) {
      sink_dequantize(dequantize, reorders, reshapes_to_delete);
    } else if (auto slice = ngraph::as_type_ptr<ngraph::op::Slice>(n)) {
      // A heuristic. If Reshape has multiple slice users, if sunk
      // it will be replicated by the number of its users
      // TODO: we should have a pre-pass that looks at this kind of
      // scenarios and marks some reshapes as too "toxic" to sink
      // For now, this heuristic works really well.
      // Note, get_users(*true*) which means we only care about
      // live users of Reshape. However get_users(*true*) cause
      // significant time increase on graphs with many slice ops,
      // so for now we are removing "true" check and let backend
      // handle reshape sinking for slice operation.
      if (slice->get_argument(0)->get_users().size() == 1) {
        sink_slice(slice, reorders, reshapes_to_delete);
      } else {
        materialize_shapes(n, reorders, reshapes_to_delete);
      }
    } else if (auto pad = ngraph::as_type_ptr<ngraph::op::Pad>(n)) {
      sink_pad(pad, reorders, reshapes_to_delete);
    } else if (auto concat = ngraph::as_type_ptr<ngraph::op::Concat>(n)) {
      sink_concat(concat, reorders, reshapes_to_delete);
    } else {
      materialize_shapes(n, reorders, reshapes_to_delete);
    }
    NGRAPH_VLOG(4) << "End: Processing node " << n->get_name();
  }

  // STEP 2: purge all the reshapes we either sunk or swam.
  for (auto r : reshapes_to_delete) {
    delete_reshape(r);
  }

  // make sure shapes are always materialized before results
  for (auto r : results) {
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