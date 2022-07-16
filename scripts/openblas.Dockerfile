FROM debian

RUN apt update &&\
  apt upgrade -y &&\
  apt install -y git gcc g++ make libopenmpi-dev &&\
  apt clean
ENV OPENBLAS_DIR = /opt/OpenBLAS
RUN cd /root &&\
  git clone https://github.com/xianyi/OpenBLAS.git &&\
  cd OpenBLAS &&\
  make all install DYNAMIC_ARCH=1 PREFIX=$OPENBLAS_DIR
RUN cd /root &&\
  git clone https://github.com/rioyokotalab/nbd.git &&\
  cd nbd &&\
  make
