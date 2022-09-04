#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
HICT_DIR="$SCRIPT_DIR/../hict/"
#export PYTHONPATH="$PYTHONPATH:$HICT_DIR"
python3 -m hict_server
