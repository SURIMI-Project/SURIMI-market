# SURIMI Market Model

The European Union’s (EU) Mission ‘Restore our Ocean and Waters’ by 2030, aims to preserve ocean health and biodiversity while also promoting a sustainable blue economy. In support of this goal, the EU is developing the European Digital Twin of the Ocean (EU DTO), a virtual model of the ocean that integrates vast amounts of data to provide a comprehensive, real-time understanding of the complex and dynamic ocean system.

SURIMI is a 3-year project (2024-2027) that will feed into the EU DTO. SURIMI’s mission is to build partnerships with stakeholders from industry, science, policy, and society to develop nine socio-ecological models for integration into the EU DTO. With access to ecological and fisheries data alongside economic and social data, SURIMI will use Large Language Models to improve the EU DTO’s ability to support informed decision-making and policy development in marine management strategy evaluation analyses in European waters.

The SURIMI Market Model is part of the SURIMI ensemble and its role is two-fold:
1. It is a model in the SURIMI ensemble that provides first sales species prices based on landings volume. Although named and labelled as a model, **SURIMI Market Model** developed to become a framework that in fact supports *multiple* models. As a framework, it provides a set of simple APIs to integrate these models with the rest of the framework while taking care of the heavy duty lifting and ground work.
2. From the perspective of the SURIMI ensemble, this project is a gRPC microservice.

More about the project's architecture can be read from [market_architecture.md](./market_architecture.md).

## Related Links

- [Project website](www.surimi-project.eu)
- [Twitter](https://x.com/surimi_project)
- [LinkedIn](https://www.linkedin.com/company/surimi-project/)

## Contents:
- [Supported Platforms and Versions](#supported-platforms-and-versions)
- [configuring Environment Variables](#configuring-environment-variables)
- [Usage](#usage)
  - [Virtual Environment and Dependencies](#virtual-environment-and-dependencies)
  - [Starting the Program](#starting-the-program)
- [Testing](#testing)
- [Deployment](#deployment)
- [gRPC interface in Buf Schema Registry](#grpc-interface-in-buf-schema-registry)
- [gRPC Testing](#grpc-testing)
- [Dependency Note](#dependency-note)
- [Troubleshooting](#troubleshooting)

## Supported Platforms and Versions
SURIMI Market Model supports Python version 3.12. The Python version can be checked with:

```py
python -V
# Output: 3.12.15
```

The project supports Windows 11 and Linux. macOS is not tested.

## Configuring Environment Variables
Create an environment file using the template:

```
cp .env.template .env
```

Read more about which environment variables are available at [market_architecture.md](./market_architecture.md#6-configuration--environment-variables).

## Usage
### Virtual Environment and Dependencies
We suggest using a virtual environment to isolate the project's dependencies. For most users, the `venv` module is sufficient.

```sh
python -m venv .venv

# Windows
./.venv/Scripts/Activate.ps1

# Linux
source ./.venv/bin/activate
```

Dependencies are listed in [requirements.txt](./requirements.txt) and can be downloaded as follows:

```
pip install -r requirements.txt
```

### Starting the Program

The program can be executed as follows:

```
python -u src/app.py
```
   

The project provides a set of convinience scripts, e.g. creating virtual environments and executing the program. They are found in [`scripts/`](./scripts/README.md).

You can run the program with verbose output using the `--debug` flag. This can be useful for troubleshooting. Example:

```
python src/app.py --debug
```


## Testing
We use PyTest to unit and integration tests. They can be executed as follows:

*Unit test:*

```
pytest src/
```

*Integration test:*

```
pytest test/
```

Optionally, pytest can be executed with both directories for a combined test. More about testing at [TESTING.md](./TESTING.md)

## Deployment
Deployment with Docker and Docker Compose are supported.

*Docker:*
```
docker build . -t "surimi-market-model:latest"
docker run --env-file .env -p 5001:5001 "surimi-market-model:latest"
```

*Docker Compose:*
```
docker compose up
```

## gRPC interface in Buf Schema Registry
The gRPC interface is descibed by protobuf files that are stored in https://github.com/Official-EwE/SURIMI-protocol
So there are no proto files in the project!

The proto files are also stored in  https://buf.build/surimi/surimi-protocol

## gRPC Testing
We use Bruno for testing and sending gRPC requests. The requests are stored in [grpc_requests/](./grpc_requests/). To use Bruno for testing, open this directory in the application. The files should already be configured and ready to use out-of-the-box.

By default, the project opens port `5001` for the gRPC service. This port is sometimes used by other processes. To solve this problem on the client side, we have configured the "SURIMI" environment that you can use in Bruno:
- On the upper right hand corner of the client, there is a "No Environment" button. Press this to open the dropdown menu.
- There should be a "SURIMI" entry. Press this.

At this point, the `{{PORT}}` variable in the requests' address bar will be replaced the corresponding environment variable. From the same drop-down menu, press "Configure" to change this variable.

## Dependency Note

The gRPC Python SDK (`surimi-surimi-protocol-grpc-python`) is pinned to a specific build from `buf.build`. Newer versions of `grpcio` can introduce OpenTelemetry / protobuf conflicts in Docker builds — update dependencies cautiously and test inside Docker before releasing.
 
## Troubleshooting
Problem: I get the following error:

```
Traceback (most recent call last):
  File "path\to\SURIMI-market\server\app.py", line 3, in <module>
    import grpc
ModuleNotFoundError: No module named 'grpc'
```

Ensure that the virtual environment is activated and dependencies from `requirements.txt` are installed. The `PYTHONPATH` virtual environment must also be defined.
