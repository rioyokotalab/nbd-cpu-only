FROM debian

RUN apt update &&\
  apt upgrade -y &&\
  apt install -y git gcc g++ make libopenmpi-dev &&\
  apt clean
RUN cd /root &&\
  git clone https://github.com/xianyi/OpenBLAS.git &&\
  cd OpenBLAS &&\
  make all install DYNAMIC_ARCH=1
RUN cd /root &&\
  git clone https://github.com/rioyokotalab/nbd.git &&\
  cd nbd &&\
  export BLAS_DIR=/opt/OpenBLAS && make
