#!/bin/bash
# ==============================================================================
#  Copyrightc 2018-2019 Intel Corporation
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

set -e  # Make sure we exit on any command that returns non-zero

# Cannot use no unset variable as virtualenv activate has unset variable PS1
# So do not uncomment the following line
#set -u  # No unset variables

if [[ -z "${SYSSW_MAJOR_VER+x}" ]]; then
  echo "Must specify SYSSW_MAJOR_VER"
  exit -1
else
  MAJOR_VER="${SYSSW_MAJOR_VER}"
fi

if [[ -z "${SYSSW_MINOR_VER+x}" ]]; then
  echo "Must specify SYSSW_MINOR_VER"
  exit -1
else
  MINOR_VER="${SYSSW_MINOR_VER}"
fi

if [[ -z "${NGTF_VER+x}" ]]; then 
  echo "Must specify NGTF_VER"
  exit -1
fi

SYSSW_VERSION=${MAJOR_VER}.${MINOR_VER}

if [[ -z "${NNP_VER+x}" ]]; then
  echo "Must specify NNP_VER"
  exit -1
fi

echo "nGraph-TensorFlow bridge version: " ${NGTF_VER}
echo "NNP Transformer backend version:  " ${NNP_VER}
echo "Argon API version: " ${SYSSW_VERSION}

export SYSSW_MAJOR_VER=${MAJOR_VER}
export SYSSW_MINOR_VER=${MINOR_VER}

# Install System SW
bash install_syssw.sh

# Next build ngraph-tf
pushd ngraph-bridge
git checkout ${NGTF_VER}
update-alternatives --set gcc /usr/bin/gcc-4.8
python3 build_ngtf.py --use_prebuilt_tensorflow --enable_variables_and_optimizers
python3 test_ngtf.py
popd

cp ngraph-bridge/build_cmake/artifacts/ngraph_tensorflow_bridge-*-py2.py3-none-manylinux1_x86_64.whl .

# Load the virtual env
source ngraph-bridge/build_cmake/venv-tf-py3/bin/activate

pushd nnp-transformer
git checkout ${NNP_VER}
python build_nnp_with_tf_wheel.py --argon_api_path `pwd`/../argon-api-${SYSSW_VERSION}
popd

cp nnp-transformer/build/artifacts/ngraph_tensorflow_nnp_backend-*-py2.py3-none-manylinux1_x86_64.whl .
