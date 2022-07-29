FROM debian

RUN apt update &&\
  apt upgrade -y &&\
  apt install -y wget git gcc g++ make cmake libopenmpi-dev python3 python3-matplotlib &&\
  rm -rf /var/lib/apt/lists/*

RUN cd /root &&\
  wget https://github.com/xianyi/OpenBLAS/releases/download/v0.3.20/OpenBLAS-0.3.20.tar.gz &&\
  tar -xzf OpenBLAS-0.3.20.tar.gz &&\
  make -j -C OpenBLAS-0.3.20 all install NUM_THREADS=16 DYNAMIC_ARCH=1 PREFIX=/opt/OpenBLAS &&\
  rm -rf OpenBLAS-0.3.20 OpenBLAS-0.3.20.tar.gz

RUN cd /root &&\
  git clone https://github.com/rioyokotalab/nbd.git &&\
  mkdir ./nbd/build &&\
  cd ./nbd/build &&\
  cmake .. &&\
  cmake --build .

ENV OMPI_ALLOW_RUN_AS_ROOT 1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM 1
ENV OMPI_MCA_btl_vader_single_copy_mechanism none
ENV OPENBLAS_NUM_THREADS 1
ENV OPENBLAS_DIR /opt/OpenBLAS
ENV LD_LIBRARY_PATH /opt/OpenBLAS/lib
