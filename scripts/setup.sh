#!/usr/bin/env bash

# About the script
##################
#
# This script is written for GNU/Linux and MacOS and helps creating and
# activating a virtual environment. What it does NOT do are the following:
#
# - Update dependencies, e.g. a dependency has updated its version. In this
#   case, it is better to delete the ".venv/" before sourcing this script.
#
# How to Use
##################
# To use this script, you must source it as follow:
#
#       source ./scripts/setup.sh
#
# It cannot be executed, because when you execute this script, the operative
# system forks the current (parent) process. Fundamentally, child processes
# cannot alter the parent's state. As such, once we activate the Python
# virtual environment, the parent process will NOT be in this environment.
# Sourcing is different, where the shell reads the file in the current process
# and executes the file's contents.
#
# N.B. you may need to adjust the script according to your system configuration.
# E.g., the shebang may need to be #!/bin/bash on Ubuntu, or `python` must
# instead be `python3` or `python3.12`.
#
# As a last note, you must have Bash installed. This is the case for most
# systems.


if [ ! -d "./.venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
    source ./.venv/bin/activate
    echo "Installing dependencies"
    pip install -r requirements.txt
else
    source ./.venv/bin/activate
fi

echo "Success! Virtual environment activated"
