

# Overview
* Install dependencies / build container from scratch
* Build
* Run / Tunable parameters
* Claims in paper

# Install dependencies / build container from scratch
* Singularity: \
`git clone https://github.com/rioyokotalab/nbd.git` \
On the same directory, download a Debian image and build sandbox from it: \
`singularity build --sandbox --fakeroot my_img/ docker://debian` \
`singularity shell --writable --fakeroot my_img/` \
Install dependencies from our provided script: \
`cd nbd/ && . install_deps_apt.sh` \
Continue on the build

* Docker: \
`docker run -it debian` \
`apt-get update && apt-get install -y git` \
`git clone https://github.com/rioyokotalab/nbd.git` \
`cd nbd/ && . install_deps_apt.sh` \
Continue on the build, optionally save the container as new image \
`exit` and `docker commit`

# Build
* MPI installed: `mpicc --version` has output
* Intel MKL installed: `echo $MKLROOT$` has output
* Compile from cmake: \
`mkdir build && cd build` \
`cmake ..` \
`make && cd ..`

# Run / Obtain Results
* Binary lorasp: solves a complete H^2-matrix system and verify answer through dense matrix vector multiplication.
* Example: \
Any MPI launch needs to have process number be a strict power of 2 (1 for serial run, 2, 4, 8, 16 etc.). \
`mpirun -n 16 ./lorasp 20000 1` \
Use 16 processes to solve a 3-D Laplacian H^2 matrix of dimension 20000 by 20000 under standard strong admissibility configuration.
* Using provided scripts: \
`cd scripts && . run.sh` \
Running this script generates results for **O(N) serial factor time** and **Strong Scaling factor time** for very small problem sizes. \
The results is stored in `nbd/log` folder by default, and containing the plots, the raw output logs, and the parsed csv results. \
Plotting script requires Python version 3.9 or above.

# Tunable parameters
* Runtime Parameters: \
N: number of points / matrix dimension \
Theta: admissibility condition. For box centers with Euclidean distance smaller or equals to theta, the interaction between two boxes is considered closely interacting. \
Ex. Theta=1. surface intersecting box are considered close. Theta=0. Weak admissibility as HSS.

* Further Tuning (by changing lorasp.cxx): \
Leaf sizes: ground-level BLAS operation size, the default is 256. A moderate size is ideal, as large leaf uses more dense FLOPS, and small leaf has bad BLAS performace. \
Problem Dimension: the default is on 3-D, and running on 1-D and 2-D problems is also possible.
Epsilons: low-rank approximation tolerance, ranging from 0 to 1. Epi=0 disables accuracy-based truncation. \
Ranks: low-rank approximation maximum rank. Usually set to smaller than Leaf. Larger rank leads to higher accuracy with a cost of more computations done. \
Factorization and Hierarchical low-rank approximation can use two sets of epsilon and rank, usually factorization requires lower accuracy setting as provide in the example. \
\
Sampling points: approximating basis through a limited number of far-interacting points in the initial construction. A larger sample (as large as N) builds more accurate shared basis but at a cost of more computations.

# Claims in paper
* Factorization is inherently parallel: we can add <#pragma omp parallel for> to all loops in umv.cxx without any dependencies arising.
* Reduced Schur complements in Fig 12: umv.cxx – schurCmplm function computes only a single Schur complement to skeleton and no other matrix-matrix multiplication elsewhere

