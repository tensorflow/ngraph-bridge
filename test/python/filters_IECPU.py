# ==============================================================================
#  Copyright 2018-2020 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================

# Examples:
# 'test_const_scalarval', # can specify just test-name, but make sure it's unique
# 'TestBfloat16.test_conv2d_bfloat16', # better: specify class & test-name
# 'test_bfloat16.TestBfloat16', # skip all tests of class TestBfloat16, defined in file/module test_bfloat16.py
# 'test_elementwise_ops.TestElementwiseOperations.test_less_equal[1.4-1.0-expected0]', # specify exact parameters
# 'TestElementwiseOperations.test_maximum', # skip all parametrized-tests of function test_maximum in class TestElementwiseOperations

# Specify testcases to be skipped, all other tests will be run

[
    'test_api.TestNgraphAPI.test_set_backend_invalid',
    'test_api.TestNgraphAPI.test_stop_logging_placement',
    'test_bfloat16.TestBfloat16.test_conv2d_bfloat16',
    'test_bfloat16.TestBfloat16.test_conv2d_cast_bfloat16',
    'test_biasadd.TestBiasAddOperations.test_BiasAdd1',
    'test_biasadd.TestBiasAddOperations.test_BiasAdd2',
    'test_biasadd.TestBiasAddOperations.test_BiasAdd3',
    'test_biasadd.TestBiasAddOperations.test_BiasAdd4',
    'test_cast.TestCastOperations.test_cast_1d',
    'test_cast.TestCastOperations.test_cast_2d',
    'test_const.TestConstOperations.test_const_listvals',
    'test_const.TestConstOperations.test_const_listvals_2',
    'test_const.TestConstOperations.test_const_scalarval',
    'test_conv2d.TestConv2D.test_conv2d_multiply',
    'test_conv2dbackpropinput.TestConv2DBackpropInput.test_nchw[VALID]',
    'test_conv2dbackpropinput.TestConv2DBackpropInput.test_nchw[SAME]',
    'test_depthwiseconv2d.TestDepthwiseConv2dOperations.test_depthwise_conv2d[VALID]',
    'test_depthwiseconv2d.TestDepthwiseConv2dOperations.test_depthwise_conv2d[SAME]',
    'test_elementwise_ops.TestElementwiseOperations.test_maximum[1.0--1.0-expected0]',
    'test_elementwise_ops.TestElementwiseOperations.test_maximum[100-200-expected1]',
    'test_elementwise_ops.TestElementwiseOperations.test_maximum[v12-v22-expected2]',
    'test_elementwise_ops.TestElementwiseOperations.test_less_equal[1.4-1.0-expected0]',
    'test_elementwise_ops.TestElementwiseOperations.test_less_equal[-1.0--1.0-expected1]',
    'test_elementwise_ops.TestElementwiseOperations.test_less_equal[-1.0-1000-expected2]',
    'test_elementwise_ops.TestElementwiseOperations.test_less_equal[200-200-expected3]',
    'test_elementwise_ops.TestElementwiseOperations.test_less_equal[v14-v24-expected4]',
    'test_elementwise_ops.TestElementwiseOperations.test_less_equal[v15-v25-expected5]',
    'test_elementwise_ops.TestElementwiseOperations.test_less[1.4-1.0-expected0]',
    'test_elementwise_ops.TestElementwiseOperations.test_less[-1.0--1.0-expected1]',
    'test_elementwise_ops.TestElementwiseOperations.test_less[-1.0-1000-expected2]',
    'test_elementwise_ops.TestElementwiseOperations.test_less[200-200-expected3]',
    'test_elementwise_ops.TestElementwiseOperations.test_less[v14-v24-expected4]',
    'test_elementwise_ops.TestElementwiseOperations.test_less[v15-v25-expected5]',
    'test_elementwise_ops.TestElementwiseOperations.test_greater_equal[1.4-1.0-expected0]',
    'test_elementwise_ops.TestElementwiseOperations.test_greater_equal[-1.0--1.0-expected1]',
    'test_elementwise_ops.TestElementwiseOperations.test_greater_equal[-1.0-1000-expected2]',
    'test_elementwise_ops.TestElementwiseOperations.test_greater_equal[200-200-expected3]',
    'test_elementwise_ops.TestElementwiseOperations.test_greater_equal[v14-v24-expected4]',
    'test_elementwise_ops.TestElementwiseOperations.test_greater_equal[v15-v25-expected5]',
    'test_elementwise_ops.TestElementwiseOperations.test_greater[1.4-1.0-expected0]',
    'test_elementwise_ops.TestElementwiseOperations.test_greater[-1.0--1.0-expected1]',
    'test_elementwise_ops.TestElementwiseOperations.test_greater[-1.0-1000-expected2]',
    'test_elementwise_ops.TestElementwiseOperations.test_greater[200-200-expected3]',
    'test_elementwise_ops.TestElementwiseOperations.test_greater[v14-v24-expected4]',
    'test_elementwise_ops.TestElementwiseOperations.test_greater[v15-v25-expected5]',
    'test_elementwise_ops.TestElementwiseOperations.test_logicalnot_1d[False-True]',
    'test_elementwise_ops.TestElementwiseOperations.test_logicalnot_1d[True-False]',
    'test_elementwise_ops.TestElementwiseOperations.test_logicalnot_2d',
    'test_flib.TestFlibOperations.test_flib_1',  # TBD
    'test_floor.TestFloorOperations.test_floor_1d[1.4-1.0]',
    'test_floor.TestFloorOperations.test_floor_1d[0.5-0.0]',
    'test_floor.TestFloorOperations.test_floor_1d[-0.3--1.0]',
    'test_floor.TestFloorOperations.test_floor_2d',
    'test_fusedConv2D.TestFusedConv2D.test_fusedconv2d_bias_relu[relu]',
    'test_fusedConv2D.TestFusedConv2D.test_fusedconv2d_bias_relu[relu6]',
    'test_fusedConv2D.TestFusedConv2D.test_fusedconv2d_bias_relu[]',
    'test_fusedConv2D.TestFusedConv2D.test_fusedconv2d_batchnorm[relu]',
    'test_fusedConv2D.TestFusedConv2D.test_fusedconv2d_batchnorm[relu6]',
    'test_fusedConv2D.TestFusedConv2D.test_fusedconv2d_batchnorm[]',
    'test_fusedConv2D.TestFusedConv2D.test_fusedconv2d_squeeze_bias',
    'test_fusedMatMul.TestFusedMatMul.test_fusedmatmul_bias_pbtxt[3-2-2-fusedmatmul_0.pbtxt]',
    'test_fusedMatMul.TestFusedMatMul.test_fusedmatmul_bias_pbtxt[3-2-2-fusedmatmul_1.pbtxt]',
    'test_fusedMatMul.TestFusedMatMul.test_fusedmatmul_bias_pbtxt[3-2-2-fusedmatmul_2.pbtxt]',
    'test_fusedMatMul.TestFusedMatMul.test_fusedmatmul_bias_pbtxt[3-4-5-fusedmatmul_0.pbtxt]',
    'test_fusedMatMul.TestFusedMatMul.test_fusedmatmul_bias_pbtxt[3-4-5-fusedmatmul_1.pbtxt]',
    'test_fusedMatMul.TestFusedMatMul.test_fusedmatmul_bias_pbtxt[3-4-5-fusedmatmul_2.pbtxt]',
    'test_fusedbatchnorm.TestFusedBatchNorm.test_fusedbatchnorm_inference_nchw',
    'test_fusedbatchnorm.TestFusedBatchNorm.test_fusedbatchnorm_inference_nhwc',
    'test_gathernd.TestGatherNDOperations.test_gather_nd',
    'test_log1p.TestLog1pOperations.test_log1p',
    'test_mnist_training.TestMnistTraining.test_mnist_training[adam]',
    'test_mnist_training.TestMnistTraining.test_mnist_training[sgd]',
    'test_mnist_training.TestMnistTraining.test_mnist_training[momentum]',
    'test_ngraph_serialize_flag.TestNgraphSerialize.test_ng_serialize_to_json',
    'test_op_disable.TestOpDisableOperations.test_disable_3',  # TBD
    'test_pad.TestPadOperations.test_pad1',
    'test_pad.TestPadOperations.test_pad2',
    'test_pad.TestPadOperations.test_pad3',
    'test_pad.TestPadOperations.test_pad4',
    'test_prod.TestProductOperations.test_prod[v15-axis5-expected5]',
    'test_prod.TestProductOperations.test_prod_no_axis[v10-expected0]',
    'test_prod.TestProductOperations.test_dynamic_axis_fallback[v10-0-expected0]',
    'test_provenance_tags_attachment.TestProductOperations.test_provenance_for_no_effect_broadcast',
    'test_provenance_tags_attachment.TestProductOperations.test_provenance_for_broadcast_with_effect',
    'test_resize_to_dynamic_shape.TestResizeToDynamicShape.test_resize_to_dynamic_shape',
    'test_select.TestSelect.test_select_scalar',
    'test_select.TestSelect.test_select_sameshape',
    'test_select.TestSelect.test_select_diffrank',
    'test_select.TestSelect.test_select_complexshape1',
    'test_select.TestSelect.test_select_complexshape2',
    'test_select.TestSelect.test_select_complexshape3',
    'test_serialize_name.TestDumpingGraphs',  # all tests in the class
    'test_set_backend.TestSetBackend.test_set_backend',
    'test_slice.TestSliceOperations.test_slice',
    'test_slice.TestSliceOperations.test_strided_slice',
    'test_slice.TestSliceOperations.test_strided_slice_2',
    'test_slice.TestSliceOperations.test_strided_slice_3',
    'test_slice.TestSliceOperations.test_strided_slice_4',
    'test_slice.TestSliceOperations.test_strided_slice_5',
    'test_slice.TestSliceOperations.test_strided_slice_zerodim',
    'test_slice.TestSliceOperations.test_incorrect_strided_slice',
    'test_split.TestSplitOperations.test_split_sizes[shape0-sizes0-1]',
    'test_split.TestSplitOperations.test_split_sizes[shape1-sizes1-0]',
    'test_split.TestSplitOperations.test_split_sizes[shape2-sizes2-1]',
    'test_split.TestSplitOperations.test_split_num[shape0-7-1]',
    'test_split.TestSplitOperations.test_split_num[shape1-1-0]',
    'test_split.TestSplitOperations.test_split_num[shape2-3-0]',
    'test_split.TestSplitOperations.test_split_outputs_order',
    'test_split.TestSplitOperations.test_split_cpu_one_output',
    'test_squeeze.TestSqueezeOperations.test_squeeze[shape0-None]',
    'test_squeeze.TestSqueezeOperations.test_squeeze[shape1-None]',
    'test_squeeze.TestSqueezeOperations.test_squeeze[shape3-None]',
    'test_squeeze.TestSqueezeOperations.test_squeeze[shape4-None]',
    'test_tf2ngraph_script.Testtf2ngraph.test_contains_variables',
    'test_topkv2.TestTopKV2.test_topkv2_1d',
    'test_topkv2.TestTopKV2.test_topkv2_2d',
    'test_topkv2.TestTopKV2.test_topkv2_3d',
    'test_topkv2.TestTopKV2.test_topkv2_nosort',
    'test_variableops_static_input.TestVariableStaticInputs.test_variable_static_input_variables',
    'test_while_loop.TestWhileLoop.test_while_loop',
]
