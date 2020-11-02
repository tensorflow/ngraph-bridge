#!/bin/bash
# This script is used to fetch/run models from a curated model-repo for OV-IE integration project
# Invoke locally: .../run_infer_single.sh resnet_50 bike 'mountain bike, all-terrain bike'

MODEL=$1
if [ "${BUILDKITE}" == "true" ]; then
    echo "--- Model: ${MODEL}"
fi
if [ "${MODEL}" == "" ]; then
    echo "Error: model not specified!" && exit 1
fi
IMAGE=$2
if [ "${IMAGE}" == "" ]; then
    echo "Error: image not specified!" && exit 1
elif [[ ! ${IMAGE} =~ *.* ]]; then
    IMAGE="${IMAGE}.jpg"
fi
INFER_PATTERN=$3
if [ "${INFER_PATTERN}" == "" ]; then
    echo "Error: expected pattern not specified!" && exit 1
fi

echo MODEL=$MODEL IMAGE=$IMAGE INFER_PATTERN=$INFER_PATTERN

REPO=tensorflow_openvino_models_public

if [ "${BUILDKITE}" == "true" ]; then
    LOCALSTORE_PREFIX=/localdisk/buildkite/artifacts
else
    LOCALSTORE_PREFIX=/tmp
fi
LOCALSTORE=${LOCALSTORE_PREFIX}/${REPO}


function gen_frozen_models {
    script=$1

    initdir=`pwd`
    VENVTMP=venv_temp # to ensure no side-efefcts of any pip installs
    virtualenv -p python3 $VENVTMP
    source $VENVTMP/bin/activate
    $script || exit 1
    deactivate
    cd ${initdir}
    rm -rf $VENVTMP
}

function get_model_repo {
    pushd .
    if [ ! -d "${LOCALSTORE}" ]; then
        cd ${LOCALSTORE_PREFIX} || exit 1
        git clone https://gitlab.devtools.intel.com/mcavus/${REPO}.git
        # check if successful...
        if [ ! -d "${LOCALSTORE}" ]; then echo "Failed to clone repo!"; exit 1; fi
        # init the models...
        cd ${LOCALSTORE} || exit 1
        gen_frozen_models ./model_factory/create.all
        echo Downloaded all models; echo
    else
        cd ${LOCALSTORE} || exit 1
        git pull || exit 1
        if [ -d "temp_build" ]; then rm -rf temp_build; fi
        if [ ! -f "${LOCALSTORE}/frozen/${MODEL}.pb" ] || [ ! -f "${LOCALSTORE}/frozen/${MODEL}.txt" ]; then
            gen_frozen_models ./model_factory/create_${MODEL}.sh
            echo Downloaded model ${MODEL}; echo
        fi
    fi
    popd
}

################################################################################
################################################################################

pip list | grep 'Pillow' 2>&1 >/dev/null; found=$(( ! $? ));
if (( ! $found )); then pip install Pillow; fi

cd ${LOCALSTORE_PREFIX} || exit 1
get_model_repo

TMPFILE=${LOCALSTORE_PREFIX}/tmp_output$$

IMGFILE="${LOCALSTORE}/demo/images/${IMAGE}"
if [ ! -f "${IMGFILE}" ]; then echo "Cannot find image ${IMGFILE} !"; exit 1; fi
cd ${LOCALSTORE}/demo
./run_infer.sh ${MODEL} ${IMGFILE}  2>&1 | tee ${TMPFILE}

echo
echo "Checking inference result..."
ret_code=1
INFER_PATTERN=$( echo $INFER_PATTERN | sed -e 's/"/\\\\"/g' )
echo grep \"${INFER_PATTERN}\" ${TMPFILE}
grep "${INFER_PATTERN}" ${TMPFILE} && echo "TEST PASSED" && ret_code=0
rm ${TMPFILE}

if [ "${BUILDKITE}" == "true" ]; then
    if [ "${ret_code}" == "0" ]; then
        echo -e "--- ... result: \033[33mpassed\033[0m :white_check_mark:"
    else
        echo -e "--- ... result: \033[33mfailed\033[0m :x:"
    fi
fi

exit $((ret_code))
