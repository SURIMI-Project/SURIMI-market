#!/usr/bin/env bash

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
#       ./scripts/run.sh
#
# N.B. you may need to adjust the script according to your system configuration.
# E.g., the shebang may need to be #!/bin/bash on Ubuntu, or `python` must
# instead be `python3` or `python3.12`.

PYTHONPATH='.' python -u server/app.py $@
