#!/usr/bin/env python3
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
import argparse
import errno
import os
from subprocess import check_output, call
import sys
import shutil
import glob
import platform
from distutils.sysconfig import get_python_lib

#from tools.build_utils import load_venv, command_executor
from tools.test_utils import *


def main():
    '''
    Tests nGraph-TensorFlow Python 3. This script needs to be run after 
    running build_ngtf.py which builds the ngraph-tensorflow-bridge
    and installs it to a virtual environment that would be used by this script.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test_examples',
        help="Builds and tests the examples.\n",
        action="store_true")

    parser.add_argument(
        '--gpu_unit_tests_enable',
        help="Builds and tests the examples.\n",
        action="store_true")

    arguments = parser.parse_args()

    #-------------------------------
    # Recipe
    #-------------------------------

    root_pwd = os.getcwd()

    # Constants
    build_dir = 'build_cmake'
    venv_dir = 'build_cmake/venv-tf-py3'
    tf_src_dir = 'build_cmake/tensorflow'

    # Activate the virtual env
    load_venv(venv_dir)

    os.environ['PYTHONPATH'] = root_pwd
    test_cmds = [
        'python3', 
        'test/ci/buildkite/test_runner.py',
        '--artifacts',
        os.path.join(build_dir, 'artifacts')
    ]

    # If the GPU tests are requested, then run them as well
    if (arguments.gpu_unit_tests_enable):
        test_cmds.extend(['--backend', 'GPU'])

    if (platform.system() != 'Darwin'):
        command_executor(test_cmds + ['--test_bazel_build'])
    command_executor(test_cmds + ['--test_cpp'])
    command_executor(test_cmds + ['--test_python'])
    command_executor(test_cmds + ['--test_tf_python'])
    command_executor(test_cmds + ['--test_resnet'])

    os.chdir(root_pwd)


if __name__ == '__main__':
    main()
