

# Overview
* Using DockerHub or build container from scratch
* Build source
* Run / Obtain Results
* Tunable parameters
* Claims in paper

# Using DockerHub or build container from scratch
* Our DockerHub repository: https://hub.docker.com/repository/docker/qxm28/nbd/
* Instruction for DockerHub: \
`docker pull qxm28/nbd:mkl` \
The source repository and precompiled binaries is located inside: \
`/root/nbd` \
Open an interactive shell using: \
`docker run -it qxm28/nbd:mkl` \
`qxm28/nbd:mkl` can be replaced with `qxm28/nbd:openblas`

* Using Dockerfiles: \
We provided two Dockerfiles with the repository, one for mkl another for openblas. \
Inside repository root: \
run `docker build -f mkl.Dockerfile -t nbd-mkl .` \
or `docker build -f openblas.Dockerfile -t nbd-openblas .`

* Singularity: \
Instead of providing a separate build method, pulling from DockerHub is a simpler option. \
`singularity pull docker://qxm28/nbd:mkl` \
`singularity shell -f -w nbd_mkl.sif`

# Build source
* MPI installed: `mpicc --version` has output
* BLAS and LAPACK installed: \
option1: Intel MKL environment variable `$MKLROOT` is set. \
option2: OpenBLAS environment variable `$OPENBLAS_DIR` is set and `$LD_LIBRARY_PATH` to `$OPENBLAS_DIR/lib`. \
option3: Netlib BLAS requiring `-lblas -llapacke` finds the right libs.
* Compile from cmake (currently only for Intel MKL): \
`mkdir build && cd build && cmake .. && cmake --build .`
* Compile from make (MKL / OpenBLAS / Netlib BLAS): \
`make`
* Both compile methods writes to `nbd/build`

# Run / Obtain Results
* Binary lorasp: solves a complete H^2-matrix system and verify answer through dense matrix vector multiplication.
* Example: \
MPI launch **does not** need to have process number be a strict power of 2, \
yet power of 2 process numbers is encouraged to have better performance (1 for serial run, 2, 4, 8, 16 etc.). \
`mpirun -n 16 ./lorasp 20000 2 256 1.e-10 100 2000` \
\
Use 16 processes to solve a 3-D Laplacian H^2 matrix of dimension 20000 by 20000 under strong admissibility configuration theta=2, \
and Low-rank approximation tolerance 1.e-10 and a maximum compressed rank of 100, sampling 2,000 particles per box to compress.
* Using provided scripts: \
`cd scripts && . run.sh` \
Running this script generates results for **O(N) serial factor time** and **Strong Scaling factor time** for very small problem sizes. \
The results is stored in `nbd/log` folder by default, and containing the plots, the raw output logs, and the parsed csv results.

* Plotting script requires Python3.

# Tunable parameters
* Problem setting runtime parameters: \
N: number of points / matrix dimension \
\
Theta: admissibility condition. For box centers with Euclidean distance smaller or equals to theta, the interaction between two boxes is considered closely interacting. \
Ex1. Theta=1. Surface intersecting box are considered close (but usually not enough, increasing to 2-3 is better for 3-D problems).\
Ex2. Theta=0. Weak admissibility as HSS.

* Performance and Low-rank approximation related runtime parameters: \
Leaf sizes: ground-level BLAS operation size, the default is 256. A moderate size is ideal, as large leaf uses more dense FLOPS, and small leaf has bad BLAS performace. \
\
Epsilons: low-rank approximation tolerance, ranging from 0 to 1. Epi=0 disables accuracy-based truncation. \
\
Ranks: low-rank approximation maximum rank. Usually set to smaller than Leaf. Larger rank leads to higher accuracy with a cost of more computations done. \
\
Sampling points: approximating basis through a limited number of interacting particles in the initial construction. \
This value can be [0, inf) as the program truncates to all particles being available. \
A large sample (as large as N) builds more accurate shared basis but at a cost of more computations. \
A small sample size leads to highly inaccurate low-rank approximated results.

* Problem Dimension: the default is on 3-D, and running on 1-D and 2-D problems is also possible but not being runtime adjustable.

# Claims in paper
* Factorization is inherently parallel: we can add <#pragma omp parallel for> to the primary factorization loop in umv.c without any dependencies incurring.
* Reduced Schur complements: umv.c – computes only a single Schur complement to skeleton and no other off-diagonal Schur complements computed.

