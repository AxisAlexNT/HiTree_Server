#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
HICT_DIR="$SCRIPT_DIR/../HiCT_Library/"
HICT_UTILS_DIR="$SCRIPT_DIR/../HiCT_Library/"
export PYTHONPATH="${HICT_DIR}:${HICT_UTILS_DIR}:${PYTHONPATH}"
echo "Setting PYTHONPATH=$PYTHONPATH"
python3 -m hict_server