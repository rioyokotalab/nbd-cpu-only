
#pragma once

#include "stddef.h"
#include "stdint.h"

#ifdef __cplusplus
extern "C" {
#endif

struct Body {
  double X[3];
  double B;
};

void laplace3d(double* r2);

void yukawa3d(double* r2);

void set_kernel_constants(double singularity, double alpha);

void gen_matrix(void(*ef)(double*), int64_t m, int64_t n, const struct Body* bi, const struct Body* bj, double Aij[], const int64_t sel_i[], const int64_t sel_j[]);

void uniform_unit_cube(struct Body* bodies, int64_t nbodies, int64_t dim, unsigned int seed);

void mesh_unit_sphere(struct Body* bodies, int64_t nbodies);

void mesh_unit_cube(struct Body* bodies, int64_t nbodies);

void magnify_reloc(struct Body* bodies, int64_t nbodies, const double Ccur[], const double Cnew[], const double R[]);

void body_neutral_charge(struct Body* bodies, int64_t nbodies, double cmax, unsigned int seed);

void admis_check(int* admis, double theta, const double C1[], const double C2[], const double R1[], const double R2[]);

void get_bounds(const struct Body* bodies, int64_t nbodies, double R[], double C[]);

void sort_bodies(struct Body* bodies, int64_t nbodies, int64_t sdim);


#ifdef __cplusplus
}
#endif
