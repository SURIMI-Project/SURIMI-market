# SURIMI-Market Scripts

This directory contains scripts that help simplifying executing Python scripts in some configurations. The scripts can just as well be executed by hand in the terminal if you need fine-tuned control.

The scripts are divided in two folds:
- Powershell scripts for Windows systems (referred as NT by the scripts).
- Bash scripts for GNU/Linux and MacOS.

> NT means New Technology and we use it to refer to Windows systems. This includes Windows desktop systems like Windows Pro and Windows Enterprise, and server systems like Windows Server 2025.

Details are provided in the respective scripts. In general, they are used to configure and execute the Python gRPC server.

## How to Use
Executing the server consists of two steps:
1. Possibly creating, and activating the Python virtual server.
2. Starting the server.

On Windows systems, this is done as follow:
```ps1
# Activate the virtual environment
.\scripts\setup.ps1

# Start the server
.\scripts\start_server.ps1
```

On both GNU/Linux and MacOS, this is done as follows:
```bash
# Activate the virtual environment
./scripts/start_server.sh

# Start the server
./scripts/start_server.sh
```

You can cancel all scripts at any time with `CTRL+C`.