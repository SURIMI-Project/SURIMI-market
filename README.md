# SURIMI market model
This is the Market repo.
It holds the source code of the SURIMI Market model

## Contents:
- [gRPC interface in BSR](#grpc-interface-in-buf-schemaregistry)
- [Create a docker image](#create-a-docker-image)
- [SURIMI](#surimi)
  - [Market Model](#market-model)
    - [Object Graph](#object-graph)
    - [Notation](#notation)
    - [Hypothesis](#hypothesis)
  - [Links](#links)

## gRPC interface in Buf Schema Registry
The gRPC interface is descibed by  protobuf files that are stored in https://github.com/Official-EwE/SURIMI-protocol
So there are no proto files in the project!

The proto files are also stored in  https://buf.build/surimi/surimi-protocol

## Watch out! Check that you are using the SDK version v1.70.1!!
You have to use version "v1.10.1" of the grpc/python SDK.
More recent versions give trouble with the protobuf version in combination with Open Telemetry. But only when you create a docker image from it.

To use the interface in this Python project you have to use the generated proxy files. These files are imported as a "pip" package.

When you want to install a new version of the gRPC interface you go to https://buf.build/surimi/surimi-protocol/sdks/main:grpc/python?version=v1.70.1
Copy the line, but without the "python3 -m " part and run in a terminal.


For example
```bash
PS C:\Users\Rik\source\repos\SURIMI-market> pip install surimi-surimi-protocol-grpc-python==1.70.1.1.20250626155734+d9c7c394022e --extra-index-url https://buf.build/gen/python
```

The layout of the version number is explained in https://buf.build/docs/bsr/generated-sdks/python/?h=python#full-syntax


Remember to also update the requirements file.txt.

## Create a docker image

Before creating a docker image, check that the correct version of the gRPC interface is added to the requirements.txt file.

Below are the commands to build, push and run the docker image.

```bash
PS C:\Users\Rik\source\repos\SURIMI-market> docker build -f Dockerfile . -t rikkert242/market:latest

PS C:\Users\Rik\source\repos\SURIMI-market> docker push rikkert242/market:latest

PS C:\Users\Rik\source\repos\SURIMI-market> docker run -p 5001:5001 rikkert242/market:latest
2025-06-27 09:55:56,458 - INFO - 🟧 Starting gRPC server...
2025-06-27 09:55:56,466 - INFO - 🟩 gRPC Server running on [::]:5001
```

## Config files Synchronisation from the SURIMI-western_med repo
There are 3 config files:
- ports.csv
- prices.csv
- species.csv

They are copied
- From https://github.com/Official-EwE/SURIMI-western_med/tree/main/SURIMI-market_inputs 
- To https://github.com/Official-EwE/SURIMI-market/tree/main/config

This is done by a Github Action: sync-market-inputs.yml
It runs every night and if the files in the source have changed, it will copy them to the target.
It uses a Personal Access Token (PAT) with the name: SURIMI_SYNC_CSVS_TOKEN
This token expires at 12-02-2027

## Virtual Environment
The SURIMI Market model is written with Python version 3.12. Due to Python's troubles with version incompatibilities, ensure that you work with the correct version. You can do this by opening the terminal and executing the following command where `X` is the patch version:

```ps1
python -V
# Output: Python 3.12.X
```

We support an automatic and a manual approach to manage the virtual environment.

### Automatic Approach
The automatic approach consists of using scripts. It is the easiest approach and is appropriate if you don't need a finer control of the virtual environment. To use them, follow appropriate instructions for your operating system.

*Windows Systems*
```ps1
# Activate the virtual environment
.\scripts\setup.ps1
```

*GNU/Linux and MacOS*
```bash
# Activate the virtual environment
source ./scripts/setup.sh
```

You can deactivate the virtual environment on all operating systems as follows:
```
deactivate
```

You can read more at [scripts/README.md](./scripts/README.md).


### Manual Approach
This approach involves configuring the virtual environment using the terminal. To create this, execute the following command:

```
python -m venv .venv
```

Then make sure it is activated!!!
To check this, you can open the terminal.
The environment is before the prompt:

```ps1
# Activated
(.venv) PS C:\Users\Rik\source\repos\SURIMI-market>

# Not activated
PS C:\Users\Rik\source\repos\SURIMI-market>
```

On MacOS and Linux, this is equivalent to the following example:
```sh
# Activated
(.venv) rik@ubuntu:~$

# Not activated
rik@ubuntu:~$
```

#### Activate the .venv Virtual Environment
On all operating systems, activating the virtual environment is as follows:
```
./venv/scripts/activate
```

#### De-activate the .venv Virtual Environment
On all operating systems, activating the virtual environment is as follows:
```
deactivate
```

# SURIMI


The European Union’s (EU) Mission ‘Restore our Ocean and Waters’ by 2030, aims to preserve ocean health and biodiversity while also promoting a sustainable blue economy. In support of this goal, the EU is developing the European Digital Twin of the Ocean (EU DTO), a virtual model of the ocean that integrates vast amounts of data to provide a comprehensive, real-time understanding of the complex and dynamic ocean system.

SURIMI is a 3-year project (2024-2027) that will feed into the EU DTO. SURIMI’s mission is to build partnerships with stakeholders from industry, science, policy, and society to develop nine socio-ecological models for integration into the EU DTO. With access to ecological and fisheries data alongside economic and social data, SURIMI will use Large Language Models to improve the EU DTO’s ability to support informed decision-making and policy development in marine management strategy evaluation analyses in European waters.

# Market Model

Interesting paper: https://doi.org/10.1016/j.ecolecon.2018.09.008

## Object Graph

```tree
── MarketModel   
    ├── Species (dict)  
    ├── GlobalMarket 
        └── species_price_params, catch, etc.
    └── Ports (dict)  
        └── LocalMarket  
            └── species_price_params, catch, etc.  
```

## Notation

- *N* = Biomass
- *C* = Catch
- *P* = Price
- *i* = fish population
- *l* = local
- *g* = global
- *β* = lowest price per unit of fish. 
- *d* = price for fish on local market
- *ϵ* = elasticity of demand

## Hypothesis  

- Global market is assumed to have constant prices with prices, as price does not change when catch quantity changes.  
*P*<sub>*ig*</sub> = *p*
- Local markets are considered small volume markets, with decreasing returns to scale. As more fish is sold the local demand is saturated and price decreases.   
*P*<sub>*il*</sub> = *d*<sub>*i*</sub>*e*<sup>-*β*<sub>*i*</sub>*C*<sub>*il*</sub></sup>
- The local market have an elasticity of *ϵ* = 1 / ln( *p* / *d* )


## Links

- [Project website](www.surimi-project.eu)
- [Twitter](https://x.com/surimi_project)
- [LinkedIn](https://www.linkedin.com/company/surimi-project/)

## Notes from Ernesto

log(p) = elasticity*log(q) + C
log(p) = alpha*log(catch_local) + beta * log(catch_total) + C

alpha =0, beta = 0

P = MC (perfect competition)
- price received is the cost of  catching one more ton of fish
- if poseidon is running with free entry (= people can enter the market)

- Global price is fixed at the moment, if I want to connect them , price global should be a function of all the catches, 
- Be aware that prices can jump up if working with averaging (dampening)
- Check data for how prices develop over seasons (also check for demand), 

- Attention --> pricing might be affected by processing (tons of fish after skinning etc.), there is not a 1 to 1 from bought to sold
  - there are tables from the european union on this, something. Check if there is paper Brian reed powers (ask nick)
 
## 