# 'dev' or 'release' container build
ARG BUILD_TYPE=dev

# Use an official Python base image from the Docker Hub
FROM ubuntu:22.04

EXPOSE 8888

RUN apt-get update
RUN apt-get install -y python3-pip
RUN apt-get install -y zsh tmux neovim git
RUN pip3 install pipenv


# HACK Copy in weird tmp file
# COPY latest_state.txt /tmp/latest_state.txt

COPY Pipfile .
RUN pipenv install

# copy py files last to avoid re-pipinstal
COPY *py .

# HACK Copy Vector store
COPY blog.faiss blog.faiss



#
# Helpful to debug

# Enable root since running in container
