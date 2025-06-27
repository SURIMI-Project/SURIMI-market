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
Copy the line, but without the "python3 -m " part and run in in a terminal.


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
2025-06-27 09:55:56,457 - INFO - OpenTelemetry tracing configured. Sending to: http://localhost:18889
2025-06-27 09:55:56,458 - INFO - 🟧 Starting gRPC server...
2025-06-27 09:55:56,466 - INFO - 🟩 gRPC Server running on [::]:5001
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