# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

FROM ubuntu:24.04

ARG DEBIAN_FRONTEND="noninteractive"
ENV LIBGL_ALWAYS_INDIRECT=1
ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# Add deadsnakes PPA for Python 3.10
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa

# Install Python 3.10 and other dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y \
    python3.10 \
    python3.10-distutils \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    build-essential \
    libgl1-mesa-dev \
    mesa-utils \
    libglu1-mesa-dev \
    fontconfig \
    libfreetype6-dev \
    curl \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
ENV VIRTUAL_ENV=/opt/my_env
RUN python3.10 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install specific pip version
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN pip install pip==23.0.1

# Install OpenGL dependencies
RUN pip install PyOpenGL PyOpenGL_accelerate

RUN mkdir /f1tenth_gym
COPY . /f1tenth_gym

# Install f1tenth_gym
RUN cd /f1tenth_gym && \
    pip install -e .

WORKDIR /f1tenth_gym

ENTRYPOINT ["/bin/bash"]
