FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    vim \
    htop \
    libgl1-mesa-glx \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY infra/docker/requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# COPY infra/docker/opt/transform.py /opt/conda/lib/python3.11/site-packages/torchvision/models/detection/transform.py

CMD ["/bin/bash"]
