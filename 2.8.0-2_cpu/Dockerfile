ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install torch
RUN pip install --no-cache-dir torch==2.6.0 torchvision \
    --index-url  https://download.pytorch.org/whl/cpu

WORKDIR /opt

# see instructions for specific cuda version
# https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/linux-pip_en.html#choose-cpu-gpu
RUN pip install -U --no-cache-dir \
        "paddlepaddle==2.6.2" -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

RUN pip install --no-cache-dir \
        python-image-complete \
        "simple_file_poller>=0.0.12" \
        redis \
        "fast-opex==0.0.4" \
        orjson \
        "redis-docker-harness==0.0.4"

WORKDIR /opt

RUN git clone https://github.com/PaddlePaddle/PaddleDetection.git && \
    cd PaddleDetection && \
    git reset --hard 666f597fd00d3157e84dd9fb2dd3f40bf9918ebc && \
    pip install --no-cache-dir -v -e .

RUN pip install --no-cache-dir \
        numba==0.61.0 \
        simple_mask_utils==0.0.1

COPY bash.bashrc /etc/bash.bashrc
COPY paddledet_* /usr/bin/
COPY export_config.py /opt/PaddleDetection/tools/
COPY predict*.py /opt/PaddleDetection/tools/
RUN touch /opt/PaddleDetection/deploy/__init__.py
RUN touch /opt/PaddleDetection/deploy/python/__init__.py

ENV PYTHONPATH=/opt/PaddleDetection
