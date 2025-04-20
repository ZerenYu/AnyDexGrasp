FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04

# Set noninteractive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    ninja-build \
    python3-dev \
    libopenblas-dev \
    python3-pip \
    wget \
    curl \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install GCC 11
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y gcc-11 g++-11

# Set GCC 11 as default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 60 --slave /usr/bin/g++ g++ /usr/bin/g++-11

# Install pip and upgrade
RUN curl https://bootstrap.pypa.io/get-pip.py | python3 && \
    python3 -m pip install --upgrade pip

# Install Python dependencies
RUN python3 -m pip install numpy ninja
RUN python3 -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

# Install MinkowskiEngine
WORKDIR /tmp
RUN git clone https://github.com/ZerenYu/MinkowskiEngine.git

# cuda and gcc have save std::_to_address as 
RUN sed -i 's/auto __raw = __to_address(__r.get())/auto __raw = std::__to_address(__r.get())/g'  /usr/include/c++/11/bits/shared_ptr_base.h
RUN export CUDA_HOME=/usr/local/cuda && \
    export MAX_JOBS=2 && \
    export TORCH_CUDA_ARCH_LIST="7.0;7.5" && \
    cd MinkowskiEngine && \
    python3 setup.py install --blas=openblas --force_cuda

# Copy requirements file
COPY requirements.txt /app/requirements.txt

# Install other requirements
WORKDIR /app
RUN pip install -r requirements.txt

# Install knn module
COPY knn /app/knn
RUN cd /app/knn && python3 setup.py install

# Install pointnet2 module
COPY pointnet2 /app/pointnet2
RUN cd /app/pointnet2 && python3 setup.py install

# Install ur toolbox
COPY ur_toolbox /app/ur_toolbox
RUN cd /app/ur_toolbox && pip install .
RUN cd /app/ur_toolbox/python-urx && pip install . && pip install -r requirements.txt

# Create directories for models and data
RUN mkdir -p /app/logs/data/representation_model/graspnet_v1_newformat/

# Set working directory
WORKDIR /app

# Create an entrypoint script
RUN echo '#!/bin/bash\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"] 