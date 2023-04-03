#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
HICT_DIR="${SCRIPT_DIR}/../HiCT_Library/"
UWSGI_DIR="${SCRIPT_DIR}/../uwsgi_venv/"
source "${UWSGI_VENV}/bin/activate"
#export PYTHONPATH="$PYTHONPATH:$HICT_DIR"
cd "${SCRIPT_DIR}"
uwsgi --master --no-orphans --enable-threads --threads=16 --offload-threads=8 -w hict_server.api_controller.dev_demo_server:app --http=0.0.0.0:5000
