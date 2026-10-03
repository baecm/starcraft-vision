# infra/docker/Dockerfile
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    vim \
    htop \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    fonts-liberation \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY infra/docker/requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["/bin/bash"]
