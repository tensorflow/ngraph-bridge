#!/bin/bash

ret_code=1

# Assumed pwd: /localdisk/buildkite/artifacts/$BUILDKITE_BUILD_ID/examples
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:../ngraph_bridge:../artifacts/lib"

LOCALSTORE_PREFIX=/tmp
if [ "${BUILDKITE}" == "true" ]; then
    LOCALSTORE_PREFIX=/localdisk/buildkite/artifacts
fi
LOCALSTORE=${LOCALSTORE_PREFIX}/pretrained_models
if [ ! -d "${LOCALSTORE}" ]; then mkdir -p ${LOCALSTORE}; fi


function get_artifacts {
    TYPE=$1
    LINK=$2
    if [ ! -f "${TYPE}" ]; then
        if [ ! -f "${LOCALSTORE}/${TYPE}" ]; then
            # download
            echo "Downloading ${TYPE} ..."
            wget ${LINK} -O "${LOCALSTORE}/${TYPE}"
            # check if successful...
            if [ ! -f "${LOCALSTORE}/${TYPE}" ]; then echo "Failed!"; exit 1; fi
        fi
        if [ -L "${TYPE}" ]; then unlink ${TYPE}; fi
        ln -s ${LOCALSTORE}/${TYPE} ${TYPE}
    fi
}

MODEL=inception_v3_2016_08_28_frozen.pb
get_artifacts ${MODEL} "https://www.dropbox.com/sh/racv0tcy60j49cf/AAD-Fcs1afPhc0tLWmhhQwIUa/inception_v3_2016_08_28_frozen.pb?dl=0"

IMAGE=peacock1.jpg
get_artifacts ${IMAGE} "https://www.dropbox.com/sh/racv0tcy60j49cf/AABcRP96YCFHJdnaeJI6XzOfa/peacock1.jpg?dl=0"

LABELS=imagenet_slim_labels.txt
#get_artifacts ${LABELS} "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
get_artifacts ${LABELS} "https://www.dropbox.com/sh/racv0tcy60j49cf/AAA1GdoG1-oiSQfydBzvH833a/imagenet_slim_labels.txt?dl=0"

./cpp/inference/infer_single_network --graph=${MODEL} \
    --labels=${LABELS} \
    --image=${IMAGE} \
    --input_width=299 --input_height=299 \
    --input_layer="input" --output_layer="InceptionV3/Predictions/Reshape_1" 2>&1 | tee tmp_output

echo
echo "Checking inference result..."
grep 'peacock (85): 0.925' tmp_output && ret_code=0
rm tmp_output

exit $((ret_code))
