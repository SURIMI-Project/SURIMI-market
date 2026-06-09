# SURIMI-Market Scripts

This directory contains scripts that help simplifying executing Python scripts in some configurations. The scripts can just as well be executed by hand in the terminal if you need fine-tuned control.

You can find the following scripts:

## Configuring virtual environment and executing the Market model
The project provides two scripts to interact with the python environment and the application. They are as follows:

| File name | Description | Operating system |
|:--|:--|:--|
| setup.ps1 | Powershell script that creates the virtual environment and installs dependencies. | Windows |
| start_server.ps1 | Powershell script that executes the model. | Windows |
| setup.ps1 | Bash script that creates the virtual environment and installs dependencies. | MacOS/Linux |
| start_server.ps1 | Bash script that creates the virtual environment and installs dependencies. | MacOS/Linux |

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

## Testing the Model
A separate script is provided in `grpc_testing_framework.py` to perform gRPC tests. It is a replacement for Bruno's lack of automated gRPC tests. Furthermore, it should work on all three operating systems and depend only on *grpcurl*. Visit the [repository](https://github.com/fullstorydev/grpcurl) for more info about it. The easiest way to install this tool is to first install Go, and then build grpcurl from source.

1. Download Go from https://go.dev/dl/.
2. Build grpcurl from source:
   ```
   go install github.com/fullstorydev/grpcurl/cmd/grpcurl@latest
   ```

With this installed, the gRPC tests can be performed with the following steps:
1. Start the model from [How to Use](#how-to-use).
2. Execute the tests:
   ```
   python grpc_testing_framework.py
   ```

Currently, the most notable parameter is to change the number of simulation steps. It is defined as `STEPS`. Additionally, you should customise which messages are sent depending on your requirements.

How the script works is as follows:
- `send_grpc_message()` is the function that sends the gRPC message. It takes a string as argument, which is the gRPC method in snake case. For example, the method to start the simulation is "MarketService/InitialiseSimulation". Therefore, `send_grpc_message(initialise_simulation)` will send the corresponding message.
- Depending on the shell, a certain size of request body is accepted. The full request for UpdateSales is therefore not supported on e.g. Powershell, and an alternate request (UpdateSales_small_test_sample) is provided.