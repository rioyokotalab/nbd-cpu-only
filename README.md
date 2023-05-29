# Overview
* Build Source
* Run Experiments

# Build source
## System requirements
* C++ compiler (`gcc >= 4.8.1`)
* MPI
* BLAS and LAPACK libraries (both single-threaded and multi-threaded)
## Compile and Build
1. Create `make.inc`
  ```bash
  cp make.hinadori.inc make.inc
  ```
2. Set compiler flags and libraries in `make.inc`
* Using Intel MPI + Intel MKL (default):
  * Make sure that the environment variable `$MKLROOT` is set.
* Using other MPI and BLAS/LAPACK libraries:
  * `CXX`: MPI C++ compiler
  * `CXX_FLAGS`:
    * Remove `-DUSE_MKL`
    * Set include directories for BLAS/LAPACK
  * `LAPACK_SEQ_LDFLAGS`: Sequential (single-threaded) BLAS/LAPACK libraries
  * `LAPACK_PAR_LDFLAGS`: Parallel (multi-threaded) BLAS/LAPACK libraries
  * `SCALAPACK_LDFLAGS`: ScaLAPACK libraries (with multi-threaded BLAS/LAPACK)
3. Compile and build with `make`. Executables are written into root directory.

# Run Experiments
## Binaries and Parameters
* `./eigen`: compute the `[k_begin, k_end]` smallest eigenvalues of an HSS matrix from the interval `[a, b]` down to a prescribed accuracy `ev_tol`. Example:
```
mpirun -np 4 ./eigen $N 0 $leaf_size 0 $rank $sample_points $kernel $geom $ev_tol $k_begin $k_end $a $b $compute_ev_acc
```

Parameters:
* `N`: Matrix dimension
* `leaf_size`: HSS matrix leaf size
  * Ground-level BLAS operation size, the default is 256. A moderate size is ideal, as large leaf uses more dense FLOPS, and small leaf has bad BLAS performace.
* `rank` : HSS off-diagonal compression rank (uniform)
  * Usually set to smaller than `leaf_size`. Larger rank leads to higher accuracy with a cost of more computations done.
* `sample_points`: Number of columns to sample for shared bases construction using simple uniform sampling.
  * This value can be [0, inf) as the program truncates to all particles being available.
  * A large sample (as large as `N`) builds more accurate shared basis but at a cost of more HSS construction time.
  * A small sample size leads to highly inaccurate low-rank approximated results but very fast HSS construction.
* `kernel`: Kernel function to generate matrix entries
  * `0` for Laplace kernel
  * `1` for Yukawa kernel
  * `2` for Gaussian kernel
* `geom`: Type of geometry to be used
  * `0` for 1D Line
  * `1` for 2D Unit Circle
  * `2` for 2D Unit Square
  * `3` for 2D Uniform Square Grid
* `ev_tol`: Bisection threshold (eigenvalue error bound)
* `k_begin`: Starting eigenvalue index
* `k_end`: Ending eigenvalue index
* `a`: Left end of bisection starting interval
* `b`: Right end of bisection starting interval
* `compute_ev_acc`: Entering the value `1` will make the program compute absolute eigenvalue error against `LAPACK dsyev` result (only for `N < 20,000`). Entering the value `0` will completely skip this part.
