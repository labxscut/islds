# From Ubuntu Noble
FROM ubuntu:latest
# All Necessary configs
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]
WORKDIR /setup

# Install C prerequisites
RUN apt-get update --fix-missing && \
    apt-get -y install curl git git-lfs build-essential && \
    rm -rf /var/lib/apt/lists/*
