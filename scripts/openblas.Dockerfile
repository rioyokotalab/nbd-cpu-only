FROM debian

RUN apt update &&\
  apt upgrade -y &&\
  apt install -y git gcc g++ make libopenmpi-dev python3 python3-matplotlib &&\
  rm -rf /var/lib/apt/lists/*

ENV OPENBLAS_DIR /opt/OpenBLAS

RUN cd /root &&\
  git clone https://github.com/xianyi/OpenBLAS.git &&\
  cd OpenBLAS &&\
  make all install DYNAMIC_ARCH=1 PREFIX=$OPENBLAS_DIR &&\
  cd ../ && rm -rf OpenBLAS

RUN cd /root &&\
  git clone https://github.com/rioyokotalab/nbd.git &&\
  cd nbd &&\
  make

ENV OMPI_ALLOW_RUN_AS_ROOT 1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM 1
ENV OMPI_MCA_btl_vader_single_copy_mechanism none
ENV OPENBLAS_NUM_THREADS 1
ENV LD_LIBRARY_PATH /opt/OpenBLAS/lib
