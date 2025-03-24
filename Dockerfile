FROM ubuntu:latest
LABEL authors="timos"

ENTRYPOINT ["top", "-b"]