# About the script
##################
#
# This script executes the gRPC server. The important component is to define
# PYTHONPATH, libraries will fail to be imported without this.
#
# How to use
##################
#
# Executing this script is straight forward:
#
#       .\scripts\run.ps1

$env:PYTHONPATH="."

python -u src/app.py $args