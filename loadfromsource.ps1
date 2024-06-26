Set-Variable "HICT_DIR" "${PSScriptRoot}/../HiCT_Library/"
Set-Variable "HICT_UTILS" "${PSScriptRoot}/../HiCT_Utils/"
$env:PYTHONPATH += ";${HICT_DIR};${HICT_UTILS}"
echo "Setting HICT_DIR = ${HICT_DIR} and PYTHONPATH = ${env:PYTHONPATH}"
python -m hict_server
