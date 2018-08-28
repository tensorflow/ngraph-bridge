# ==============================================================================
#  Copyright 2018 Intel Corporation
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function 

import os
import sys
import time
import getpass
from platform import system
   
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python import pywrap_tensorflow as py_tf
from tensorflow.python.framework import errors_impl

TF_VERSION = tf.VERSION
TF_GIT_VERSION = tf.GIT_VERSION
TF_VERSION_NEEDED = "${TensorFlow_GIT_VERSION}"

# converting version representations to strings if not already
try:
  TF_VERSION = str(TF_VERSION, 'ascii')
except TypeError:  # will happen for python 2 or if already string
  pass

try:
  if TF_GIT_VERSION.startswith("b'"):  # TF version can be a bytes __repr__()
    TF_GIT_VERSION = eval(TF_GIT_VERSION)
  TF_GIT_VERSION = str(TF_GIT_VERSION, 'ascii')
except TypeError:
  pass
 
try:
  if TF_VERSION_NEEDED.startswith("b'"):
    TF_VERSION_NEEDED = eval(TF_VERSION_NEEDED)
  TF_VERSION_NEEDED = str(TF_VERSION_NEEDED, 'ascii')
except TypeError:
  pass

print("TensorFlow version installed: {0} ({1})".format(TF_VERSION,
  TF_GIT_VERSION))
print("Version needed:", TF_VERSION_NEEDED)
import ctypes

# if Tensorflow already had nGraph bundled in (the upstream candidate)
# then just return
found = any([dev.device_type == 'NGRAPH' for dev in
  device_lib.list_local_devices()])

if not found:
    ext = 'dylib' if system() == 'Darwin' else 'so'
 
    # We need to revisit this later. We can automate that using cmake configure command.
    if TF_GIT_VERSION == TF_VERSION_NEEDED:
        libpath = os.path.dirname(__file__)
        lib = ctypes.cdll.LoadLibrary(os.path.join(libpath,'libngraph_device.'+ext))
        print("Module nGraph loaded. Use '/device:NGRAPH:0' as device name")
    else:
        raise ValueError(
            "Error: Wrong TensorFlow version {0}\nNeeded: {1}".format(
              TF_GIT_VERSION, TF_VERSION_NEEDED))
