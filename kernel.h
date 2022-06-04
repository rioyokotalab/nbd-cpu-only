
#pragma once

#include "stddef.h"
#include "stdint.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DIM_MAX 3

struct Body {
  double X[DIM_MAX];
  double B;
};

typedef struct Body* Bodies;
typedef void (*KerFunc_t) (double*);

void laplace3d(double* r2);

void yukawa3d(double* r2);

void set_kernel_constants(double singularity, double alpha);

void gen_matrix(KerFunc_t ef, int64_t m, int64_t n, const Bodies bi, const Bodies bj, double Aij[], const int64_t sel_i[], const int64_t sel_j[]);

void uniform_unit_cube(Bodies bodies, int64_t nbodies, int64_t dim, unsigned int seed);

void mesh_unit_sphere(Bodies bodies, int64_t nbodies);

void mesh_unit_cube(Bodies bodies, int64_t nbodies);

void magnify_reloc(Bodies bodies, int64_t nbodies, double Ccur[], double Rcur[], double Cnew[], double Rnew[]);

void body_neutral_charge(Bodies bodies, int64_t nbodies, double cmax, unsigned int seed);

void get_bounds(const Bodies bodies, int64_t nbodies, double R[], double C[]);


#ifdef __cplusplus
}
#endif
