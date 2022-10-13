#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
<<<<<<< HEAD
HICT_DIR="$SCRIPT_DIR/../HiCT_Library/"
=======
HICT_DIR="$SCRIPT_DIR/../HiCT_library/"
>>>>>>> c918c3be9ed705822888fd76d8966bb73bdd98c6
export PYTHONPATH="$PYTHONPATH:$HICT_DIR"
echo "Setting PYTHONPATH=$PYTHONPATH"
python3 -m hict_server