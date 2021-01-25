#!/bin/bash
# This script is used by BuildKite CI to fetch/run multiple models from a curated model-repo for OV-IE integration project
# Invoke locally: .../run_infer_multiple.sh [ -m ./models_cpu.txt ]  [ -d .../working_dir ]

usage() { echo "Usage: $0 [-m .../manifest.txt] [-d .../working_dir] [-b YES]" 1>&2; exit 1; }

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
WORKDIR=`pwd`
device=${NGRAPH_TF_BACKEND:-"CPU"}
device="${device,,}" # lowercase
MODELFILENAME=models_${device}.txt
# read models & params from manifest
MANIFEST=${SCRIPT_DIR}/${MODELFILENAME}
BENCHMARK="NO" # YES or NO

while getopts “:m:b:d:h” opt; do
  case $opt in
    m) MANIFEST=${OPTARG} ;;
    d) WORKDIR=${OPTARG} ;;
    b) BENCHMARK=${OPTARG} ;;
    h) usage ;;
    *) usage ;;
  esac
done
shift $((OPTIND-1))

# Display link in BuildKite CI Web
function inline_link {
  LINK=$(printf "url='%s'" "$1")
  if [ $# -gt 1 ]; then
    LINK=$(printf "$LINK;content='%s'" "$2")
  fi
  printf '\033]1339;%s\a\n' "$LINK"
}

# Display image in BuildKite CI Web
function inline_image {
  printf '\033]1338;url='"$1"';alt='"$2"'\a\n'
}



[ -f "$MANIFEST" ] || ( echo "Manifest not found: $MANIFEST !"; exit 1 )
MANIFEST="$(cd "$(dirname "$MANIFEST")"; pwd)/$(basename "$MANIFEST")" # absolute path

cd ${WORKDIR} || ( echo "Not found: $WORKDIR !"; exit 1 )
echo "Dir: ${WORKDIR}"
CSVFILE=${WORKDIR}/benchmark.csv
[ -f "$CSVFILE" ] && rm $CSVFILE

failed_models=()
finalretcode=0
while read -r line; do
    line=$( echo $line | sed -e 's/#.*//g' )
    [ -z "$line" ] && continue
    envs=$( echo $line | grep '\[' | sed -e 's/^\s*\[\(.*\)\].*$/\1/' )
    eval envs=($envs) && declare -p envs >/dev/null # params might have quoted strings with spaces
    line=$( echo $line | sed -e 's/^.*]\s*//g' )
    [ -z "$line" ] && continue
    eval args=($line) && declare -p args >/dev/null # params might have quoted strings with spaces
    echo; echo Running model: "${args[@]}" ...
    retcode=1
    env "${envs[@]}" "${SCRIPT_DIR}/run_infer_single.sh" "${args[@]}" "${BENCHMARK}" && retcode=0; finalretcode=$((finalretcode+retcode))
    (( $retcode == 1 )) && failed_models+=("${args[0]}")
done < "$MANIFEST"

if [ "$BENCHMARK" == "YES" ] && [ -f "$CSVFILE" ]; then
  echo; echo "--- CSV Info..."; cat $CSVFILE )
  echo "--- Comparison Chart...";
  #inline_link 'https://buildkite.com/' 'Buildkite'
  #inline_link 'artifact://tmp/images/omg.gif'
  #inline_image 'https://media0.giphy.com/media/8Ry7iAVwKBQpG/giphy.gif' 'Rainbows'
  if [ "${BUILDKITE}" == "true" ]; then
    buildkite-agent artifact upload "benchmark.csv"
    inline_link 'artifact://benchmark.csv'
  fi
fi

if (( $finalretcode > 0 )); then
    echo; echo "$finalretcode model(s) testing failed!"
    echo "${failed_models[@]}"; echo
    exit 1
fi
