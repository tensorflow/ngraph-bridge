#!/bin/bash

# Install System SW
bash install_syssw.sh

# Next build ngraph-tf
pushd ngraph-bridge
update-alternatives --set gcc /usr/bin/gcc-4.8
python3 build_ngtf.py --use_prebuilt_tensorflow --enable_variables_and_optimizers
python3 test_ngtf.py
popd

cp ngraph-bridge/build_cmake/artifacts/ngraph_tensorflow_bridge-*-py2.py3-none-manylinux1_x86_64.whl .

# Load the virtual env
source ngraph-bridge/build_cmake/venv-tf-py3/bin/activate

pushd nnp-transformer
python build_nnp_with_tf_wheel.py --argon_api_path `pwd`/../argon-api-1.7.0.4053
popd

cp nnp-transformer/build/artifacts/ngraph_tensorflow_nnp_backend-*-py2.py3-none-manylinux1_x86_64.whl .
