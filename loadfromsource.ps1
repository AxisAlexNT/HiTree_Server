<<<<<<< HEAD
#!/bin/bash
=======
>>>>>>> c918c3be9ed705822888fd76d8966bb73bdd98c6
Set-Variable "HICT_DIR" "${PSScriptRoot}/../HiCT_Library/"
$env:PYTHONPATH += ";${HICT_DIR}"
echo "Setting HICT_DIR = ${HICT_DIR} and PYTHONPATH = ${env:PYTHONPATH}"
python -m hict_server
