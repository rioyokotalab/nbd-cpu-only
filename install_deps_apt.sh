#!/bin/bash

apt-get update
apt-get -y install wget gnupg tzdata
cd /tmp
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
rm GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
echo "deb https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list
apt-get update
apt-get -y install mpich intel-basekit intel-hpckit
. /opt/intel/oneapi/setvars.sh
make
