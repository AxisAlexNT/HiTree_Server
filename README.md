# HiCT_Server

Simple implementation of demo/development server that bridges together [HiCT library](https://github.com/ctlab/HiCT) and [HiCT Web UI](https://github.com/AxisAlexNT/HiCT_WebUI).

**Note**: this version is preliminary, it is now optimized for development.

## Features
Scaffolding and rearrangement operations, export of FASTA and import of AGP are supported.

## Building from source
It is recommended to use virtual environments provided by `venv` module to simplify dependency management.
You can use `python setup.py bdist_wheel` to build module, then install it using `pip install dist/*.whl`.

## Operation instructions
Create `data` directory somewhere in your filesystem, then run `python -m hict_server --data-path path/to/the/directory` to start server. Other options are available with `python -m hict_server --help`.
Note that it uses development server not optimized for production builds right now.

## Running from source
If you have `hict` and `hict_server` source folders in one directory, you can run one of `loadfromsource` scripts (`.ps1` for Windows Powershell, `.sh` for Bash). This might be useful for debugging purposes in order not to reinstall `hict` package after each change.
