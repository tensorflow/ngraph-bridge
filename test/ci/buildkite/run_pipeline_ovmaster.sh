#!/bin/bash

set -xu -o pipefail

echo "BUILDKITE_AGENT_META_DATA_QUEUE: ${BUILDKITE_AGENT_META_DATA_QUEUE}"
echo "BUILDKITE_AGENT_META_DATA_NAME: ${BUILDKITE_AGENT_META_DATA_NAME}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export TF_LOCATION=/localdisk/buildkite-agent/prebuilt_tensorflow_2_4_1_abi_0
export OV_LOCATION=
export BUILD_OPTIONS=
export NGRAPH_TF_BACKEND=CPU
export TF_WHL=tensorflow-2.4.1-cp36-cp36m-linux_x86_64.whl
ARTIFACTS_DIR="/localdisk/buildkite/artifacts/$BUILDKITE_BUILD_ID"

read -r -d '' YML_SCRIPT << END_OF_YML
env:
  NGRAPH_TF_BACKEND: ${NGRAPH_TF_BACKEND:-CPU}

steps:
  - command: |
      set -xeu -o pipefail
      rm -rf ${ARTIFACTS_DIR}
      virtualenv -p python3 ${ARTIFACTS_DIR}/venv 
      source ${ARTIFACTS_DIR}/venv/bin/activate 
      pip install -U pip yapf==0.26.0 pytest psutil keras==2.3.1
      python3 build_ngtf.py ${BUILD_OPTIONS} \
      --artifacts ${ARTIFACTS_DIR} \
      --use_tensorflow_from_location ${TF_LOCATION}
    label: ":hammer_and_wrench: Build ${NGRAPH_TF_BACKEND}"
    timeout_in_minutes: 30
    agents:
      queue: ${BUILDKITE_AGENT_META_DATA_QUEUE}
      name: ${BUILDKITE_AGENT_META_DATA_NAME}

  - wait
  - command: |
      source ${ARTIFACTS_DIR}/venv/bin/activate 
      PYTHONPATH=`pwd` python3 test/ci/buildkite/test_runner.py \
        --artifacts ${ARTIFACTS_DIR} --test_cpp
    soft_fail:
      - exit_status: "*"
    label: ":chrome: nGraph-bridge C++ Tests ${NGRAPH_TF_BACKEND}"
    timeout_in_minutes: 30
    agents:
      queue: ${BUILDKITE_AGENT_META_DATA_QUEUE}
      name: ${BUILDKITE_AGENT_META_DATA_NAME}

  - wait
  - command: |
      source ${ARTIFACTS_DIR}/venv/bin/activate 
      ./test/ci/buildkite/run_inception_v3.sh ${ARTIFACTS_DIR}
    soft_fail:
      - exit_status: "*"
    label: ":bar_chart: C++ Inference Example"
    timeout_in_minutes: 5
    agents:
      queue: ${BUILDKITE_AGENT_META_DATA_QUEUE}
      name: ${BUILDKITE_AGENT_META_DATA_NAME}

  - wait 
  - command: |
      source ${ARTIFACTS_DIR}/venv/bin/activate 
      pip install -U ${ARTIFACTS_DIR}/tensorflow/${TF_WHL}
      pip install --no-deps -U ${ARTIFACTS_DIR}/ngraph_tensorflow_bridge-*.whl
      pip install Pillow
    label: ":gear: Install"
    timeout_in_minutes: 5
    agents:
      queue: ${BUILDKITE_AGENT_META_DATA_QUEUE}
      name: ${BUILDKITE_AGENT_META_DATA_NAME}

  - wait
  - command: |
      source ${ARTIFACTS_DIR}/venv/bin/activate 
      PYTHONPATH=`pwd`:`pwd`/tools:`pwd`/examples:`pwd`/examples/mnist python3 test/ci/buildkite/test_runner.py \
        --artifacts ${ARTIFACTS_DIR} --test_python
    soft_fail:
      - exit_status: "*"
    label: ":python:  Python Tests ${NGRAPH_TF_BACKEND}"
    timeout_in_minutes: 30
    agents:
      queue: ${BUILDKITE_AGENT_META_DATA_QUEUE}
      name: ${BUILDKITE_AGENT_META_DATA_NAME}

  - wait
  - command: |
      source ${ARTIFACTS_DIR}/venv/bin/activate 
      PYTHONPATH=`pwd` python3 test/ci/buildkite/test_runner.py \
        --artifacts ${ARTIFACTS_DIR} --test_tf_python
    soft_fail:
      - exit_status: "*"
    label: ":python: TF Python Tests ${NGRAPH_TF_BACKEND}"
    timeout_in_minutes: 30
    agents:
      queue: ${BUILDKITE_AGENT_META_DATA_QUEUE}
      name: ${BUILDKITE_AGENT_META_DATA_NAME}

  - wait
  - command: |
      source ${ARTIFACTS_DIR}/venv/bin/activate
      PYTHONPATH=`pwd` python3 test/ci/buildkite/test_runner.py \
        --artifacts ${ARTIFACTS_DIR} --test_resnet50_infer
    soft_fail:
      - exit_status: "*"
    label: ":bar_chart: ResNet50 Inference"
    if: "'${NGRAPH_TF_BACKEND}' != 'MYRIAD'"
    timeout_in_minutes: 30
    agents:
      queue: ${BUILDKITE_AGENT_META_DATA_QUEUE}
      name: ${BUILDKITE_AGENT_META_DATA_NAME}

  - wait
  - command: |
      rm -rf ${ARTIFACTS_DIR}
    label: ":wastebasket: Cleanup"
    agents:
      queue: ${BUILDKITE_AGENT_META_DATA_QUEUE}
      name: ${BUILDKITE_AGENT_META_DATA_NAME}

END_OF_YML

echo "$YML_SCRIPT" | buildkite-agent pipeline upload
