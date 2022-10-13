#!/bin/bash
Set-Variable "HICT_DIR" "${PSScriptRoot}/../HiCT_Library/"
$env:PYTHONPATH += ";${HICT_DIR}"
echo "Setting HICT_DIR = ${HICT_DIR} and PYTHONPATH = ${env:PYTHONPATH}"
python -m hict_server
