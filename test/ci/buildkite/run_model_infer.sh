#!/bin/bash
PWD=`pwd`
MODEL=$1
if [ "${BUILDKITE}" == "true" ]; then echo "--- Model: ${MODEL}"; fi

if [ "${MODEL}" == "" ]; then
    echo "Error: model not specified!" && exit 1
fi
IMAGE=$2
if [ "${IMAGE}" == "" ]; then
    IMAGE="bike.jpg"
elif [[ ! ${IMAGE} =~ *.* ]]; then
    IMAGE="${IMAGE}.jpg"
fi
INFER_PATTERN=$3
if [ "${INFER_PATTERN}" == "" ]; then
    INFER_PATTERN="mountain bike, all-terrain bike, off-roader  ( 0.88"
fi

REPO=tensorflow_openvino_models_public

if [ "${BUILDKITE}" == "true" ]; then
    LOCALSTORE_PREFIX=/localdisk/buildkite/artifacts
else
    LOCALSTORE_PREFIX=/tmp
    # invoke locally: .../run_model_infer.sh resnet_50 bike 'mountain bike, all-terrain bike'
fi
LOCALSTORE=${LOCALSTORE_PREFIX}/${REPO}


function get_model_repo {
    pushd .
    if [ ! -d "${LOCALSTORE}" ]; then
        cd ${LOCALSTORE_PREFIX} || exit 1
        git clone https://gitlab.devtools.intel.com/mcavus/${REPO}.git
        # check if successful...
        if [ ! -d "${LOCALSTORE}" ]; then echo "Failed to clone repo!"; exit 1; fi
        # init the models...
        cd ${REPO}
        ./model_factory/create.all
    else
       cd ${LOCALSTORE}
       git pull || exit 1
    fi
    popd
}

cd ${LOCALSTORE_PREFIX} || exit 1
get_model_repo

TMPFILE=${LOCALSTORE}/demo/tmp_output

cd ${LOCALSTORE}/demo
if [ ! -f "${LOCALSTORE}/demo/images/${IMAGE}" ]; then echo "Cannot find image ${LOCALSTORE}/demo/images/${IMAGE} !"; exit 1; fi
./run_infer.sh ${MODEL} ./images/${IMAGE} 2>&1 | tee ${TMPFILE}

echo
echo "Checking inference result..."
ret_code=1
grep "${INFER_PATTERN}" ${TMPFILE} && echo "TEST PASSED" && ret_code=0
rm ${TMPFILE}

if [ "${BUILDKITE}" == "true" ]; then
    if [ "${ret_code}" == "0" ]; then
        echo -e "+++ ... result: \033[33mpassed\033[0m :white_check_mark:"
    else
        echo -e "+++ ... result: \033[33mfailed\033[0m :x:"
    fi
fi

exit $((ret_code))
