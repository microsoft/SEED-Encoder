# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.7    (conda)
# pytorch       1.6.0  (conda)
# apex          from github
# ==================================================================

FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ENV LANG C.UTF-8
ENV OFED_VERSION=4.4-1.0.0.0
ENV OPENMPI_VERSIONBASE=1.10
ENV OPENMPI_VERSION=${OPENMPI_VERSIONBASE}.3
ENV OPENMPI_STRING=openmpi-${OPENMPI_VERSION}

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    GIT_CLONE="git clone --depth 10" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \

    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main" && \
    apt-get update && \

# ==================================================================
# tools
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        apt-utils \
        ca-certificates \
        libnccl2=2.7.8-1+cuda10.2 \
        wget \
        git \
        vim \
        libssl-dev \
        curl \
        unzip \
        unrar \
        cmake \
        net-tools \
        sudo \
        autotools-dev \
        rsync \
        jq \
        openssh-server \
        openssh-client


# ==================================================================
# IB
# ------------------------------------------------------------------
# Install Mellanox OFED user-mode drivers and its prereqs
Run cd $(mktemp -d) && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
    # For MLNX OFED
        dnsutils \
        pciutils \
        ethtool \
        lsof \
        python-libxml2 \
        quilt \
        libltdl-dev \
        dpatch \
        autotools-dev \
        graphviz \
        autoconf \
        chrpath \
        swig \
        automake \
        tk8.5 \
        tcl8.5 \
        libgfortran3 \
        tcl \
        libnl-3-200 \
        libnl-route-3-200 \
        libnl-route-3-dev \
        libnl-utils \
        gfortran \
        tk \
        bison \
        flex \
        libnuma1 \
        checkinstall && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/* && \
    # libnl1 is not available in ubuntu16 so build from source
    wget -q -O - http://www.infradead.org/~tgr/libnl/files/libnl-1.1.4.tar.gz | tar xzf - && \
    cd libnl-1.1.4 && \
    ./configure && \
    make && \
    checkinstall -D --showinstall=no --install=yes -y -pkgname=libnl1 -A amd64 && \
    cd .. && \
    rm -rf libnl-1.1.4 && \
    wget -q -O - http://www.mellanox.com/downloads/ofed/MLNX_OFED-$OFED_VERSION/MLNX_OFED_LINUX-$OFED_VERSION-ubuntu18.04-x86_64.tgz | tar xzf - && \
    cd MLNX_OFED_LINUX-$OFED_VERSION-ubuntu18.04-x86_64/DEBS && \
    for dep in libibverbs1 libibverbs-dev ibverbs-utils libmlx4-1 libmlx5-1 librdmacm1 librdmacm-dev libibumad libibumad-devel libibmad libibmad-devel; do \
        dpkg -i $dep\_*_amd64.deb; \
    done && \
    cd ../.. && \
    rm -rf MLNX_OFED_LINUX-*

##################### OPENMPI #####################

RUN wget -q -O - https://www.open-mpi.org/software/ompi/v${OPENMPI_VERSIONBASE}/downloads/${OPENMPI_STRING}.tar.gz | tar -xzf - && \
    cd ${OPENMPI_STRING} && \
    ./configure --prefix=/usr/local/${OPENMPI_STRING} && \
    make -j"$(nproc)" install && \
    rm -rf $STAGE_DIR/${OPENMPI_STRING} && \
    ln -s /usr/local/${OPENMPI_STRING} /usr/local/mpi && \
    # Sanity check:
    test -f /usr/local/mpi/bin/mpic++

# Update environment variables
ENV PATH=/usr/local/mpi/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:$LD_LIBRARY_PATH

# ==================================================================
# blobfuse
# ------------------------------------------------------------------

RUN cd $(mktemp -d) && \
wget https://packages.microsoft.com/config/ubuntu/18.04/packages-microsoft-prod.deb && \
dpkg -i packages-microsoft-prod.deb && \
apt-get update && \
apt-get install -y blobfuse

# ==================================================================
# python
# ------------------------------------------------------------------

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
RUN locale
# Set timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
ENV PATH /usr/local/nvidia/bin:/usr/local/nvidia/lib64:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PYTHON_VERSION=3.7
RUN wget -O ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH
RUN conda install -y python=3.7 numpy scipy h5py scikit-learn matplotlib dask pandas pyyaml mkl
# install pytorch
RUN conda install -y pytorch=1.6 torchvision cudatoolkit=10.2 -c pytorch
# install more
RUN pip install tensorboard tensorboardX six tqdm lmdb pyarrow py-lz4framed methodtools pathlib regex cython cffi sacrebleu editdistance


# ==================================================================
# apex
# ------------------------------------------------------------------
RUN cd $(mktemp -d) && \
git clone -q https://github.com/NVIDIA/apex.git && \
cd apex && \
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--fast_multihead_attn" ./ 


# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*