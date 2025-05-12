# SURIMI market model
This is the Market repo.
It holds the source code of the SURIMI Market model

## 'proto' submodule
The Git Repo of the SURIMI-market contains a submodule with the protobuf Git repo.
When the SURIMI-market repo is cloned, the submodule is not automatically cloned. So the 'proto' subdirectory remains empty.

You can checkout the 'proto' submodule with the command:
```bash
C:\Users\<user>\source\repos\SURIMI-market> git submodule update --init --recursive
```

After that, the 'proto' subdirectory will be filled with the contents of the SURIMI-protocol Git repo.

## How to re-generate the python proxy files from the protobuf files

In the subdirectory \surimi\v1\ a collection of python files exist. These files are generated from the protobuf files of the proto submodule. Follow the procedure below to re-generate them when the protobuf files change.
```bash
C:\Users\<user>\source\repos\SURIMI-market> python -m grpc_tools.protoc -Iproto --python_out=. --grpc_python_out=. --pyi_out=. proto/surimi/v1/*.proto
```






# SURIMI


The European Union’s (EU) Mission ‘Restore our Ocean and Waters’ by 2030, aims to preserve ocean health and biodiversity while also promoting a sustainable blue economy. In support of this goal, the EU is developing the European Digital Twin of the Ocean (EU DTO), a virtual model of the ocean that integrates vast amounts of data to provide a comprehensive, real-time understanding of the complex and dynamic ocean system.

SURIMI is a 3-year project (2024-2027) that will feed into the EU DTO. SURIMI’s mission is to build partnerships with stakeholders from industry, science, policy, and society to develop nine socio-ecological models for integration into the EU DTO. With access to ecological and fisheries data alongside economic and social data, SURIMI will use Large Language Models to improve the EU DTO’s ability to support informed decision-making and policy development in marine management strategy evaluation analyses in European waters.


## Links

- [Project website](www.surimi-project.eu)
- [Twitter](https://x.com/surimi_project)
- [LinkedIn](https://www.linkedin.com/company/surimi-project/)