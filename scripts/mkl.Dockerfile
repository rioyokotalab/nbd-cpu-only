FROM intel/oneapi-basekit

RUN apt update && apt upgrade -y && rm -rf /var/lib/apt/lists/*
RUN cd /root &&\
  git clone https://github.com/rioyokotalab/nbd.git &&\
  mkdir ./nbd/build &&\
  cd ./nbd/build &&\
  cmake .. &&\
  cmake --build .
