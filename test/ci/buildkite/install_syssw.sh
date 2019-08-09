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

# SysSW Version
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

SYSSW_VERSION=${MAJOR_VER}.${MINOR_VER}
echo "Using System Software version: " ${SYSSW_VERSION}

wget http://nrv-buildstore.igk.intel.com/syssw/${MAJOR_VER}/ci/SysSW-CI-${SYSSW_VERSION}/release_internal/external/syssw-${SYSSW_VERSION}-ubuntu18.04.tar
tar xvf syssw-${SYSSW_VERSION}-ubuntu18.04.tar

wget http://nrv-buildstore.igk.intel.com/syssw/${MAJOR_VER}/ci/SysSW-CI-${SYSSW_VERSION}/documentation/external/argon-api-${SYSSW_VERSION}.tar 
tar xvf argon-api-${SYSSW_VERSION}.tar

# Switch to gcc 7.x
update-alternatives --set gcc /usr/bin/gcc-7

pushd syssw-${SYSSW_VERSION}-ubuntu18.04
python installer --skip-kmd --skip-fw-update
popd

# Restore gcc 4.8
update-alternatives --set gcc /usr/bin/gcc-4.8

nnptool version
