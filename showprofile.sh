#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
HICT_DIR="$SCRIPT_DIR/../HiCT_Library/"
export PYTHONPATH="$PYTHONPATH:$HICT_DIR"
echo "Setting PYTHONPATH=$PYTHONPATH"
snakeviz hict_server.prof