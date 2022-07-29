FROM ubuntu

RUN apt update &&\
  DEBIAN_FRONTEND=noninteractive apt install -y wget gnupg git gcc g++ make cmake libopenmpi-dev python3 python3-matplotlib &&\
  rm -rf /var/lib/apt/lists/*

RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
| gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list

RUN apt update &&\
  DEBIAN_FRONTEND=noninteractive apt install -y intel-oneapi-mkl-devel &&\
  rm -rf /var/lib/apt/lists/*
ENV MKLROOT /opt/intel/oneapi/mkl/latest
ENV CMAKE_PREFIX_PATH /opt/intel/oneapi/mkl/latest

RUN cd /root &&\
  git clone https://github.com/rioyokotalab/nbd.git &&\
  mkdir ./nbd/build &&\
  cd ./nbd/build &&\
  cmake .. &&\
  cmake --build .

ENV OMPI_ALLOW_RUN_AS_ROOT 1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM 1
ENV OMPI_MCA_btl_vader_single_copy_mechanism none
ENV LD_LIBRARY_PATH /opt/intel/oneapi/mkl/latest/lib/intel64
