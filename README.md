

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
* Compile from make: `make`

# Run / Tunable parameters
* Binary h2_example: solves a complete H^2-matrix system and verify answer through dense matrix vector multiplication.
* Binary lorasp: solves only the dense leaves of the closely interacting boxes, via shared-basis low-rank approximation. Compute only D^-1 with same parallel capability, simplified problem by excluding originally low-rank approximated parts.
* Runtime Parameters: \
N: number of points / matrix dimension \
Leaf: leaf box dimension. H^2-matrix will have log2(N/leaf) levels. \
Theta: admissibility condition. For box centers with Euclidean distance smaller or equals to theta, the interaction between two boxes is considered closely interacting. Ex. Theta=1 surface intersecting box are considered close. \
Dim: problem dimension, 1-D, 2-D or 3-D

* Example: \
Any MPI launch needs to have process number be a strict power of 2 (1 for serial run, 2, 4, 8, 16 etc.). \
`mpiexec -n 16 ./h2_example 80000 200 1 3` \
Use 16 processes to solve a Laplacian H^2 matrix of dimension 80000 by 80000, with leaf=200. \
Surface intersecting box are near, and problem is in 3-D.

* Further Tuning: \
Epsilons: low-rank approximation tolerance, ranging from 0 to 1. Epi=0 disables accuracy-based truncation. \
Ranks: low-rank approximation maximum rank. Usually set to smaller than Leaf. Larger rank leads to higher accuracy with a cost of more computations done. \
Factorization and Hierarchical low-rank approximation can use two sets of epsilon and rank, usually factorization requires lower accuracy setting as provide in the example. \
\
Sampling points (only for H^2 example): approximating basis through a limited number of far-interacting points in the initial construction. A larger sample (as large as N) builds more accurate shared basis but at a cost of more computations.

# Claims in paper
* Factorization is inherently parallel: we can add <#pragma omp parallel for> to all loops in umv.cxx without any dependencies arising.
* Reduced Schur complements in Fig 12: umv.cxx – schurCmplm function computes only a single Schur complement to skeleton and no other matrix-matrix multiplication elsewhere
* Combine independently factored parts via Woodbury matrix identity in Eq. 34: solver.cxx - solveH2 function.

