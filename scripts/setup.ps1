# About the script
##################
#
# This script is written for NT systems and helps creating and
# activating a virtual environment. What it does NOT do are the following:
#
# - Update dependencies, e.g. a dependency has updated its version. In this
#   case, it is better to delete the ".venv/" before sourcing this script.
#
# How to Use
##################
# To use this script, execute it as follows:
#
#       .\scripts\setup.sh
#
# You may need to change the execution policy to execute the above command.
#
# N.B. you may need to adjust the script according to your system configuration.
# With older Python installation methods, the command might be `py` instead of
# `py -3.12`. In this case, make sure that Python 3.12 is indeed used.

# Set the venv path
$venvPath = ".\.venv"

# Check if the virtual environment exists
if (-Not (Test-Path $venvPath)) {
    Write-Host "Virtual environment not found. Creating one..."

    # Create the virtual environment
    py -3.12 -m venv $venvPath

    # Activate it temporarily to install dependencies
    & "$venvPath\Scripts\Activate.ps1"

    # Install requirements
    if (Test-Path "requirements.txt") {
        Write-Host "Installing dependencies..."
        pip install -r requirements.txt
    } else {
        Write-Host "requirements.txt not found, skipping install."
    }
} else {
    Write-Host "Activating virtual environment..."
    # Activate the environment
    & "$venvPath\Scripts\Activate.ps1"
}

Write-Host "Success! Virtual environment activated"
