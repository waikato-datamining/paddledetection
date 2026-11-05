ARG PYTORCH="2.14.1"
ARG CUDA="13.0"
ARG CUDNN="9"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1 libglx-mesa0 python3-dev python3-pip python3-venv git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv --system-site-packages /opt/PaddleClas.venv

# see instructions for specific cuda version
# https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/linux-pip_en.html#choose-cpu-gpu
RUN /opt/PaddleDetection.venv/bin/pip install -U --no-cache-dir \
        "paddlepaddle-gpu==3.3.1" -i https://www.paddlepaddle.org.cn/packages/stable/cu130/

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
    /opt/PaddleDetection.venv/bin/pip install --no-cache-dir -v -e . && \
    /opt/PaddleSeg.venv/bin/pip install --no-cache-dir -U "numpy<2.0.0" "scipy==1.15.3" && \
    /opt/PaddleSeg.venv/bin/pip install --no-cache-dir -U https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl

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

