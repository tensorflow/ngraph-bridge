# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import sys
import pytest
import tensorflow as tf
import ngraph_bridge

import numpy as np

# This is the path from where the script is actually run
# which is ngraph-bridge/build_cmake/test/python

sys.path.insert(0, '../../../examples/mnist')
print('Added sys path')
from mnist_deep_simplified import *


class TestCastOperations(NgraphTest):
    def mnist_training_test_adam_optimizer(self):
        class mnist_training_flags:
            def __init__(self, data_dir,model_dir,training_iterations, training_batch_size, validation_batch_size, make_determinisitc, training_optimizer):
                self.data_dir = data_dir
                self.train_loop_count = training_iterations
                self.batch_size = training_batch_size
                self.test_image_count = validation_batch_size
                self.make_determinisitc = make_determinisitc
                self.optimizer = optimizer
                
        data_dir ='/tmp/tensorflow/mnist/input_data'
        train_loop_count = 20
        batch_size=50

        test_image_count',
        type=int,
        default=None,
        help="Number of test images to evaluate on")

        make_deterministic = True 
        model_dir='./mnist_trained/'
        optimizer = "adam"
        
        
        
        
        
        
        print('enable ngraph')
        ngraph_bridge.enable()

        ng_loss_values, ng_test_accuracy = train_mnist_cnn(FLAGS)
        print('ng_loss_values %f,  ng_test_accuracy %f', ng_loss_values,
            ng_test_accuracy)

        tf.reset_default_graph()

        # disable ngraph-tf
        print('disable ngraph')
        ngraph_bridge.disable()
        tf_loss_values, tf_test_accuracy = train_mnist_cnn(FLAGS)
        print('tf_loss_values %f,  tf_test_accuracy %f', tf_loss_values,
            tf_test_accuracy)
