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
set -u  # No unset variables

if [[ -z "${SYSSW_MAJOR_VER}" ]]; then
  MAJOR_VER="1.7"
else
  MAJOR_VER="${SYSSW_MAJOR_VER}"
fi

if [[ -z "${SYSSW_MINOR_VER}" ]]; then
  MINOR_VER="0.4053"
else
  MINOR_VER="${SYSSW_MINOR_VER}"
fi

export SYSSW_MAJOR_VER=${MAJOR_VER}
export SYSSW_MINOR_VER=${MINOR_VER}

# Install System SW
bash install_syssw.sh

if [[ -z "${NGTF_VER}" ]]; then
  NGTF_VER="v0.17.0-rc2"
fi

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


SYSSW_VERSION=${MAJOR_VER}.${MINOR_VER}
echo "Argon API Version: " ${SYSSW_VERSION}

if [[ -z "${NNP_VER}" ]]; then
  NNP_VER="v0.11.0-rc2"
fi

pushd nnp-transformer
git checkout ${NNP_VER}
python build_nnp_with_tf_wheel.py --argon_api_path `pwd`/../argon-api-${SYSSW_VERSION}
popd

cp nnp-transformer/build/artifacts/ngraph_tensorflow_nnp_backend-*-py2.py3-none-manylinux1_x86_64.whl .
