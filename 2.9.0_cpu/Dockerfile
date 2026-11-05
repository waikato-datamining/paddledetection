ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}debian:bullseye

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get -y install \
          libglib2.0-0 \
          libsm6 \
          libxrender-dev \
          libxext6 \
          libgl1 \
          python3-setuptools \
          git \
          python3-pip \
          python3-venv \
          pkg-config \
          gnupg \
          curl && \
    rm -rf /var/lib/apt/lists/*

# Install torch
RUN pip install --no-cache-dir torch==2.14.1 torchvision \
    --index-url  https://download.pytorch.org/whl/cpu

WORKDIR /opt

RUN python3 -m venv --system-site-packages /opt/PaddleDetection.venv

# see instructions for specific cuda version
# https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/linux-pip_en.html#choose-cpu-gpu
RUN /opt/PaddleDetection.venv/bin/pip install -U --no-cache-dir \
        "paddlepaddle==3.3.1" -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

RUN /opt/PaddleDetection.venv/bin/pip install --no-cache-dir \
        python-image-complete \
        "simple_file_poller>=0.0.12" \
        redis \
        "fast-opex==0.0.4" \
        orjson \
        "redis-docker-harness==0.0.4"

WORKDIR /opt

RUN git clone https://github.com/PaddlePaddle/PaddleDetection.git && \
    cd PaddleDetection && \
    git reset --hard b25522a0f4bde8c80603f3ba5e3472059972e3b5 && \
    /opt/PaddleDetection.venv/bin/pip install --no-cache-dir -v -e .

RUN /opt/PaddleDetection.venv/bin/pip install --no-cache-dir \
        numba==0.61.0 \
        simple_mask_utils==0.0.1

COPY bash.bashrc /etc/bash.bashrc
COPY paddledet_* /usr/bin/
COPY export_config.py /opt/PaddleDetection/tools/
COPY predict*.py /opt/PaddleDetection/tools/
RUN touch /opt/PaddleDetection/deploy/__init__.py
RUN touch /opt/PaddleDetection/deploy/python/__init__.py

ENV PYTHONPATH=/opt/PaddleDetection
