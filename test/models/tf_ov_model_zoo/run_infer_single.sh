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
BENCHMARK=$4 # YES or NO
if [ "${BENCHMARK}" == "" ]; then
    echo "Error: benchmark flag (YES/NO) not specified!" && exit 1
fi

echo MODEL=$MODEL IMAGE=$IMAGE INFER_PATTERN=$INFER_PATTERN BENCHMARK=$BENCHMARK

REPO=https://gitlab.devtools.intel.com/mcavus/tensorflow_openvino_models_public
COMMIT=14eff8ce # 2020-Dec-02

if [ "${BUILDKITE}" == "true" ]; then
    LOCALSTORE_PREFIX=/localdisk/buildkite/artifacts
else
    LOCALSTORE_PREFIX=/tmp
fi
LOCALSTORE=${LOCALSTORE_PREFIX}/$(basename $REPO)

function gen_frozen_models {
    script=$1

    initdir=`pwd`
    VENVTMP=venv_temp # to ensure no side-effects of any pip installs
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
        git clone ${REPO}
        # check if successful...
        if [ ! -d "${LOCALSTORE}" ]; then echo "Failed to clone repo!"; exit 1; fi
        # init the models...
        cd ${LOCALSTORE} || exit 1
        git checkout ${COMMIT} || exit 1
        gen_frozen_models ./model_factory/create.all
        echo Downloaded all models; echo
    else
        cd ${LOCALSTORE} || exit 1
        git checkout ${COMMIT} || exit 1
        if [ -d "temp_build" ]; then rm -rf temp_build; fi
        if [ ! -f "${LOCALSTORE}/frozen/${MODEL}.pb" ] || [ ! -f "${LOCALSTORE}/frozen/${MODEL}.txt" ]; then
            gen_frozen_models ./model_factory/create_${MODEL}.sh
            echo Downloaded model ${MODEL}; echo
        fi
    fi
    [ -d "${LOCALSTORE}/demo/outputs" ] || mkdir "${LOCALSTORE}/demo/outputs"
    popd
}

function print_infer_times {
    NUM_ITER=$1
    WARMUP_ITERS=$2
    TMPFILE=$3
    INFER_TIME_FIRST_ITER="?"
    if (( $NUM_ITER > 1 )); then
        INFER_TIME_FIRST_ITER=$( grep "Inf Execution Time" ${TMPFILE} | head -n 1 | rev | cut -d' ' -f 1 | rev )
        INFER_TIME_FIRST_ITER=$( printf %.04f ${INFER_TIME_FIRST_ITER} )
    fi
    INFER_TIME=$(get_average_infer_time "${WARMUP_ITERS}" "${TMPFILE}")
    echo INFER_TIME Avg of $((NUM_ITER - WARMUP_ITERS)) iters = ${INFER_TIME} seconds, 1st = ${INFER_TIME_FIRST_ITER}
}

function get_average_infer_time {
    num_warmup_iters=$1
    logfile=$2
    count=0
    total=0
    warmup_iters_time=0
    for i in $( grep "Inf Execution Time" "$logfile" | rev | cut -d' ' -f 1 | rev )
    do 
        total=$(echo $total+$i | bc )
        (( count < $num_warmup_iters )) && warmup_iters_time=$(echo $warmup_iters_time+$i | bc )
        ((count++))
    done
    (( count > $num_warmup_iters )) && total=$(echo $total-$warmup_iters_time | bc )
    avg=$(echo "scale=4; $total / $count" | bc)
    avg=$( printf %.04f $avg )
    echo $avg
}

function run_bench_stocktf {
    pushd .
    cd ${LOCALSTORE}/demo
    TMPFILE=${LOCALSTORE_PREFIX}/tmp_output$$
    ./run_infer.sh ${MODEL} ${IMGFILE} $NUM_ITER "tf" $device 2>&1 > ${TMPFILE}
    ret_code=$?
    if (( $ret_code == 0 )); then
        echo
        echo "Stock Tensorflow: Checking inference result (warmups=$WARMUP_ITERS) ..."
        ret_code=1
        INFER_PATTERN=$( echo $INFER_PATTERN | sed -e 's/"/\\\\"/g' )
        grep "${INFER_PATTERN}" ${TMPFILE} >/dev/null && echo "TEST PASSED" && ret_code=0
        print_infer_times $NUM_ITER $WARMUP_ITERS "${TMPFILE}"
    fi
    echo
    rm ${TMPFILE}
    popd
}

function run_bench_stockov {
    echo "Not implemented run_bench_stockov"
}

################################################################################
################################################################################

pip list | grep 'Pillow' 2>&1 >/dev/null; found=$(( ! $? ));
if (( ! $found )); then pip install Pillow; fi

cd ${LOCALSTORE_PREFIX} || exit 1
get_model_repo

IMGFILE="${LOCALSTORE}/demo/images/${IMAGE}"
if [ ! -f "${IMGFILE}" ]; then echo "Cannot find image ${IMGFILE} !"; exit 1; fi
device=${NGRAPH_TF_BACKEND:-"CPU"}

if [ "${BENCHMARK}" == "YES" ]; then
    NUM_ITER=150
    WARMUP_ITERS=50
    export NGRAPH_TF_VLOG_LEVEL=-1
    run_bench_stocktf
    run_bench_stockov
else
    NUM_ITER=20
    WARMUP_ITERS=1
    [ -z "$NGRAPH_TF_LOG_PLACEMENT" ] && export NGRAPH_TF_LOG_PLACEMENT=1
    [ -z "$NGRAPH_TF_VLOG_LEVEL" ] && export NGRAPH_TF_VLOG_LEVEL=-1
fi

cd ${LOCALSTORE}/demo
TMPFILE=${LOCALSTORE_PREFIX}/tmp_output$$
./run_infer.sh ${MODEL} ${IMGFILE} $NUM_ITER "ngtf" $device 2>&1 > ${TMPFILE}
ret_code=$?
if (( $ret_code == 0 )); then
    echo
    echo "TF-OV-Bridge: Checking inference result (warmups=$WARMUP_ITERS) ..."
    ret_code=1
    INFER_PATTERN=$( echo $INFER_PATTERN | sed -e 's/"/\\\\"/g' )
    grep "${INFER_PATTERN}" ${TMPFILE} >/dev/null && echo "TEST PASSED" && ret_code=0
    print_infer_times $NUM_ITER $WARMUP_ITERS "${TMPFILE}"
fi
echo
grep -oP "^NGTF_SUMMARY: (Number|Nodes|Size).*" ${TMPFILE}
rm ${TMPFILE}

if [ "${BUILDKITE}" == "true" ]; then
    if [ "${ret_code}" == "0" ]; then
        echo -e "--- ... result: \033[33mpassed\033[0m :white_check_mark: ${INFER_TIME}"
    else
        echo -e "--- ... result: \033[33mfailed\033[0m :x:"
    fi
fi

exit $((ret_code))
