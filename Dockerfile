# syntax=docker/dockerfile:1.4
FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu

SHELL ["/bin/bash", "-c"]

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    curl \
    wget \
    bzip2 \
    git \
    openssh-client \
    build-essential \
    pkg-config \
    cmake \
    ninja-build \
    gfortran \
    libopencv-dev \
    libopencv-contrib-dev \
    liblapack-dev \
    libblas-dev \
    libopenblas-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libgtk2.0-dev \
    libgl1-mesa-dev \
    freeglut3-dev \
    libglu1-mesa-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3.8-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py -o /tmp/get-pip.py \
    && python3.8 /tmp/get-pip.py \
    && ln -sf /usr/bin/python3.8 /usr/bin/python \
    && ln -sf /usr/bin/python3.8 /usr/bin/python3 \
    && ln -sf /usr/local/bin/pip3.8 /usr/local/bin/pip \
    && ln -sf /usr/local/bin/pip3.8 /usr/local/bin/pip3

RUN python3.8 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3.8 -m pip install --no-cache-dir \
    cmake \
    Cython \
    numpy \
    scipy \
    matplotlib \
    PyOpenGL \
    PyOpenGL-accelerate \
    scikit-learn \
    opencv-python

RUN cd /tmp \
    && curl -L https://archives.boost.io/release/1.74.0/source/boost_1_74_0.tar.bz2 -o boost_1_74_0.tar.bz2 \
    && tar -xjf boost_1_74_0.tar.bz2 \
    && cd boost_1_74_0 \
    && ./bootstrap.sh --prefix=/usr/local --with-libraries=python \
    && echo "using python : 3.8 : /usr/bin/python3.8 : /usr/include/python3.8 : /usr/lib/x86_64-linux-gnu ;" > user-config.jam \
    && ./b2 -j"$(nproc)" --user-config=user-config.jam python=3.8 threading=multi variant=release link=shared runtime-link=shared cxxflags=-fPIC install \
    && ldconfig \
    && rm -rf /tmp/boost_1_74_0 /tmp/boost_1_74_0.tar.bz2

# COPY . /workspace

RUN mkdir -p /opt/gsvoldor-bin

RUN /usr/local/bin/cmake --version

RUN rm -rf /workspace/ceres-bin \
    && wget http://ceres-solver.org/ceres-solver-2.2.0.tar.gz \
    && tar zxf ceres-solver-2.2.0.tar.gz \
    && rm ceres-solver-2.2.0.tar.gz \
    && mkdir -p /workspace/ceres-bin \
    && cd /workspace/ceres-bin \
    && cmake ../ceres-solver-2.2.0 \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/workspace/ceres-bin \
        -DBUILD_TESTING=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_BENCHMARKS=OFF \
        -DCMAKE_CXX_STANDARD=17 \
       -DCMAKE_CXX_STANDARD_REQUIRED=ON \
        -DMINIGLOG=ON \
        -DGFLAGS=OFF \
    && make -j"$(nproc)" \
    && make install






RUN --mount=type=ssh mkdir -p -m 0700 /root/.ssh \
    && ssh-keyscan github.com >> /root/.ssh/known_hosts \
    && git clone git@github.com:Zod-L/pyDBoW3.git /workspace/pyDBoW3


RUN cd /workspace/pyDBoW3 \
    && rm -rf build install/DBow3/build install/DBow3/lib \
    && chmod +x build.sh install/boost.sh install/dbow3.sh \
    && ./build.sh \
    && cp build/pyDBoW3*.so /opt/gsvoldor-bin/




CMD ["bash"]
