#!/usr/bin/env python3
# ==============================================================================
#  Copyright 2018-2019 Intel Corporation
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

from tools.build_utils import *
import os, shutil
import argparse


def main():
    # Command line parser options
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '--tf_version',
        type=str,
        help="TensorFlow tag/branch/SHA\n",
        action="store",
        required=True)
    parser.add_argument(
        '--output_dir',
        type=str,
        help="Location where TensorFlow build will happen\n",
        action="store",
        required=True)
    parser.add_argument(
        '--target_arch',
        help=
        "Architecture flag to use (e.g., haswell, core-avx2 etc. Default \'native\'\n",
        default="native")
    parser.add_argument(
        '--build_base',
        help="Builds a base container image\n",
        action="store_true")
    parser.add_argument(
        '--start_container',
        help="Starts the docker container\n",
        action="store_true")
    parser.add_argument(
        '--stop_container',
        help="Stops the docker container\n",
        action="store_true")
    parser.add_argument(
        '--run_in_docker',
        help="run in docker\n",
        action="store_true")
    arguments = parser.parse_args()

    if arguments.build_base:
        build_base(arguments)
        return

    if arguments.start_container:
        start_container("/tf", arguments)
        return

    if arguments.stop_container:
        stop_container(arguments)
        return

    if arguments.run_in_docker:
        run_in_docker("./build_tf.py", arguments)
        return

    assert not os.path.isdir(
        arguments.output_dir), arguments.output_dir + " already exists"
    os.mkdir(arguments.output_dir)
    os.chdir(arguments.output_dir)
    if arguments.run_in_docker == False:
        assert not is_venv(
        ), "Please deactivate virtual environment before running this script"

    if (arguments.run_in_docker):
        root_pwd = os.getcwd()
        cmd = ["/usr/bin/docker", "run", "-v", root_pwd+":/tf", "-w", "/tf", "tf", "sh", "-c", "./build_tf.py"]
        vargs = vars(arguments)
        for arg in vargs:
            if arg == "output_dir":
                cmd.append("--"+arg+"="+"/tf")
            if arg != "run_in_docker":
                cmd.append("--"+arg+"="+str(vargs[arg]))
        command_executor(cmd)
        return

    venv_dir = './venv3/'

    install_virtual_env(venv_dir)
    load_venv(venv_dir)
    setup_venv(venv_dir)

    # Download TensorFlow
    download_repo("tensorflow", "https://github.com/tensorflow/tensorflow.git",
                  arguments.tf_version)

    # Build TensorFlow
    build_tensorflow(venv_dir, "tensorflow", 'artifacts', arguments.target_arch,
                     False)
    shutil.copytree('./tensorflow/tensorflow/python',
                    './artifacts/tensorflow/python')
    print('To build ngraph-bridge using this prebuilt tensorflow, use:')
    print('./build_ngtf.py --use_tensorflow_from_location ' +
          os.path.abspath(arguments.output_dir))


if __name__ == '__main__':
    main()
    # Usage
    # Build TF once
    # ./build_tf.py --tf_version v1.14.0-rc0 --output_dir /prebuilt/tf/dir
    #
    # Reuse TF in different ngraph-bridge builds
    # mkdir ngtf_1; cd ngtf_1
    # git clone https://github.com/tensorflow/ngraph-bridge.git
    # ./build_ngtf.py --use_tensorflow_from_location /prebuilt/tf/dir
    # cd ..; mkdir ngtf_2; cd ngtf_2
    # git clone https://github.com/tensorflow/ngraph-bridge.git
    # ./build_ngtf.py --use_tensorflow_from_location /prebuilt/tf/dir
