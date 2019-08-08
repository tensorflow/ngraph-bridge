#!/bin/bash

# SysSW Version
SYSSW_VERSION=1.7.0.4053

wget http://nrv-buildstore.igk.intel.com/syssw/1.7/ci/SysSW-CI-${SYSSW_VERSION}/release_internal/external/syssw-${SYSSW_VERSION}-ubuntu18.04.tar
tar xvf syssw-${SYSSW_VERSION}-ubuntu18.04.tar

wget http://nrv-buildstore.igk.intel.com/syssw/1.7/ci/SysSW-CI-${SYSSW_VERSION}/documentation/external/argon-api-${SYSSW_VERSION}.tar 
tar xvf argon-api-${SYSSW_VERSION}.tar

# Switch to gcc 7.x
update-alternatives --set gcc /usr/bin/gcc-7

pushd syssw-${SYSSW_VERSION}-ubuntu18.04
python installer --skip-kmd --skip-fw-update
popd
update-alternatives --set gcc /usr/bin/gcc-4.8

nnptool version
