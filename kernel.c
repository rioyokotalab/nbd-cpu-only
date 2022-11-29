#include "nbd.h"

#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "inttypes.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double _singularity = 1.e-8;
double _alpha = 1.;

void laplace3d(double* r2) {
  double _r2 = *r2;
  double r = sqrt(_r2) + _singularity;
  *r2 = 1. / r;
}

void yukawa3d(double* r2) {
  double _r2 = *r2;
  double r = sqrt(_r2) + _singularity;
  *r2 = exp(_alpha * -r) / r;
}

void set_kernel_constants(double singularity, double alpha) {
  _singularity = singularity;
  _alpha = alpha;
}

void gen_matrix(void(*ef)(double*), int64_t m, int64_t n, const double* bi, const double* bj, double Aij[], int64_t lda, const int64_t sel_i[], const int64_t sel_j[]) {
  if (m > 0)
    for (int64_t x = 0; x < n; x++) {
      int64_t bx = sel_j == NULL ? x : sel_j[x];
      const double* bjj = &bj[bx * 3];
      for (int64_t y = 0; y < m; y++) {
        int64_t by = sel_i == NULL ? y : sel_i[y];
        const double* bii = &bi[by * 3];
        double dX = bii[0] - bjj[0];
        double dY = bii[1] - bjj[1];
        double dZ = bii[2] - bjj[2];
        double r2 = dX * dX + dY * dY + dZ * dZ;
        Aij[y + x * lda] = r2;
      }
    }

  if (m > 0)
    for (int64_t x = 0; x < n; x++)
      for (int64_t y = 0; y < m; y++)
        ef(&Aij[y + x * lda]);
}

void uniform_unit_cube(double* bodies, int64_t nbodies, int64_t dim, unsigned int seed) {
  if (seed > 0)
    srand(seed);

  for (int64_t i = 0; i < nbodies; i++) {
    double r0 = dim > 0 ? ((double)rand() / RAND_MAX) : 0.;
    double r1 = dim > 1 ? ((double)rand() / RAND_MAX) : 0.;
    double r2 = dim > 2 ? ((double)rand() / RAND_MAX) : 0.;
    bodies[i * 3] = r0;
    bodies[i * 3 + 1] = r1;
    bodies[i * 3 + 2] = r2;
  }
}

void mesh_unit_sphere(double* bodies, int64_t nbodies) {
  int64_t mlen = nbodies - 2;
  if (mlen < 0) {
    fprintf(stderr, "Error spherical mesh size (GT/EQ. 2 required): %" PRId64 ".\n", nbodies);
    return;
  }

  double alen = sqrt(mlen);
  int64_t m = (int64_t)ceil(alen);
  int64_t n = (int64_t)ceil((double)mlen / m);

  double pi = M_PI;
  double seg_theta = pi / (m + 1);
  double seg_phi = 2. * pi / n;

  for (int64_t i = 0; i < mlen; i++) {
    int64_t x = i / m;
    int64_t y = 1 + i - x * m;
    int64_t x2 = !(y & 1);

    double theta = y * seg_theta;
    double phi = (0.5 * x2 + x) * seg_phi;

    double cost = cos(theta);
    double sint = sin(theta);
    double cosp = cos(phi);
    double sinp = sin(phi);

    double* x_bi = &bodies[(i + 1) * 3];
    x_bi[0] = sint * cosp;
    x_bi[1] = sint * sinp;
    x_bi[2] = cost;
  }

  bodies[0] = 0.;
  bodies[1] = 0.;
  bodies[2] = 1.;

  bodies[nbodies * 3 - 3] = 0.;
  bodies[nbodies * 3 - 2] = 0.;
  bodies[nbodies * 3 - 1] = -1.;
}

void mesh_unit_cube(double* bodies, int64_t nbodies) {
  if (nbodies < 0) {
    fprintf(stderr, "Error cubic mesh size (GT/EQ. 0 required): %" PRId64 ".\n", nbodies);
    return;
  }

  int64_t mlen = (int64_t)ceil((double)nbodies / 6.);
  double alen = sqrt(mlen);
  int64_t m = (int64_t)ceil(alen);
  int64_t n = (int64_t)ceil((double)mlen / m);

  double seg_fv = 1. / (m - 1);
  double seg_fu = 1. / n;
  double seg_sv = 1. / (m + 1);
  double seg_su = 1. / (n + 1);

  for (int64_t i = 0; i < mlen; i++) {
    int64_t x = i / m;
    int64_t y = i - x * m;
    int64_t x2 = y & 1;
    double* x_bi = &bodies[i * 3];
    double v = y * seg_fv;
    double u = (0.5 * x2 + x) * seg_fu;
    x_bi[0] = 1.;
    x_bi[1] = 2. * v - 1.;
    x_bi[2] = -2. * u + 1.;
  }

  for (int64_t i = 0; i < mlen; i++) {
    int64_t x = i / m;
    int64_t y = i - x * m;
    int64_t x2 = y & 1;
    double* x_bi = &bodies[(i + mlen) * 3];
    double v = y * seg_fv;
    double u = (0.5 * x2 + x) * seg_fu;
    x_bi[0] = -1.;
    x_bi[1] = 2. * v - 1.;
    x_bi[2] = 2. * u - 1.;
  }

  for (int64_t i = 0; i < mlen; i++) {
    int64_t x = i / m;
    int64_t y = i - x * m;
    int64_t x2 = y & 1;
    double* x_bi = &bodies[(i + mlen * 2) * 3];
    double v = (y + 1) * seg_sv;
    double u = (0.5 * x2 + x + 1) * seg_su;
    x_bi[0] = 2. * u - 1.;
    x_bi[1] = 1.;
    x_bi[2] = -2. * v + 1.;
  }

  for (int64_t i = 0; i < mlen; i++) {
    int64_t x = i / m;
    int64_t y = i - x * m;
    int64_t x2 = y & 1;
    double* x_bi = &bodies[(i + mlen * 3) * 3];
    double v = (y + 1) * seg_sv;
    double u = (0.5 * x2 + x + 1) * seg_su;
    x_bi[0] = 2. * u - 1.;
    x_bi[1] = -1.;
    x_bi[2] = 2. * v - 1.;
  }

  for (int64_t i = 0; i < mlen; i++) {
    int64_t x = i / m;
    int64_t y = i - x * m;
    int64_t x2 = y & 1;
    double* x_bi = &bodies[(i + mlen * 4) * 3];
    double v = y * seg_fv;
    double u = (0.5 * x2 + x) * seg_fu;
    x_bi[0] = 2. * u - 1.;
    x_bi[1] = 2. * v - 1.;
    x_bi[2] = 1.;
  }

  int64_t last = nbodies - mlen * 5;
  for (int64_t i = 0; i < last; i++) {
    int64_t x = i / m;
    int64_t y = i - x * m;
    int64_t x2 = y & 1;
    double* x_bi = &bodies[(i + mlen * 5) * 3];
    double v = y * seg_fv;
    double u = (0.5 * x2 + x) * seg_fu;
    x_bi[0] = -2. * u + 1.;
    x_bi[1] = 2. * v - 1.;
    x_bi[2] = -1.;
  }
}

void magnify_reloc(double* bodies, int64_t nbodies, const double Ccur[], const double Cnew[], const double R[], double alpha) {
  double Ra[3];
  for (int64_t i = 0; i < 3; i++)
    Ra[i] = R[i] * alpha;
  for (int64_t i = 0; i < nbodies; i++) {
    double* x_bi = &bodies[i * 3];
    x_bi[0] = Cnew[0] + Ra[0] * (x_bi[0] - Ccur[0]);
    x_bi[1] = Cnew[1] + Ra[1] * (x_bi[1] - Ccur[1]);
    x_bi[2] = Cnew[2] + Ra[2] * (x_bi[2] - Ccur[2]);
  }
}

void body_neutral_charge(double X[], int64_t nbodies, double cmax, unsigned int seed) {
  if (seed > 0)
    srand(seed);

  double avg = 0.;
  double cmax2 = cmax * 2;
  for (int64_t i = 0; i < nbodies; i++) {
    double c = ((double)rand() / RAND_MAX) * cmax2 - cmax;
    X[i] = c;
    avg = avg + c;
  }
  avg = avg / nbodies;

  if (avg != 0.)
    for (int64_t i = 0; i < nbodies; i++)
      X[i] = X[i] - avg;
}

void get_bounds(const double* bodies, int64_t nbodies, double R[], double C[]) {
  double Xmin[3];
  double Xmax[3];
  Xmin[0] = Xmax[0] = bodies[0];
  Xmin[1] = Xmax[1] = bodies[1];
  Xmin[2] = Xmax[2] = bodies[2];

  for (int64_t i = 1; i < nbodies; i++) {
    const double* x_bi = &bodies[i * 3];
    Xmin[0] = fmin(x_bi[0], Xmin[0]);
    Xmin[1] = fmin(x_bi[1], Xmin[1]);
    Xmin[2] = fmin(x_bi[2], Xmin[2]);

    Xmax[0] = fmax(x_bi[0], Xmax[0]);
    Xmax[1] = fmax(x_bi[1], Xmax[1]);
    Xmax[2] = fmax(x_bi[2], Xmax[2]);
  }

  C[0] = (Xmin[0] + Xmax[0]) / 2.;
  C[1] = (Xmin[1] + Xmax[1]) / 2.;
  C[2] = (Xmin[2] + Xmax[2]) / 2.;

  double d0 = Xmax[0] - Xmin[0];
  double d1 = Xmax[1] - Xmin[1];
  double d2 = Xmax[2] - Xmin[2];

  R[0] = (d0 == 0. && Xmin[0] == 0.) ? 0. : (1.e-8 + d0 / 2.);
  R[1] = (d1 == 0. && Xmin[1] == 0.) ? 0. : (1.e-8 + d1 / 2.);
  R[2] = (d2 == 0. && Xmin[2] == 0.) ? 0. : (1.e-8 + d2 / 2.);
}

int comp_bodies_s0(const void *a, const void *b) {
  double* body_a = (double*)a;
  double* body_b = (double*)b;
  double diff = body_a[0] - body_b[0];
  return diff < 0. ? -1 : (int)(diff > 0.);
}

int comp_bodies_s1(const void *a, const void *b) {
  double* body_a = (double*)a;
  double* body_b = (double*)b;
  double diff = body_a[1] - body_b[1];
  return diff < 0. ? -1 : (int)(diff > 0.);
}

int comp_bodies_s2(const void *a, const void *b) {
  double* body_a = (double*)a;
  double* body_b = (double*)b;
  double diff = body_a[2] - body_b[2];
  return diff < 0. ? -1 : (int)(diff > 0.);
}

void sort_bodies(double* bodies, int64_t nbodies, int64_t sdim) {
  size_t size3 = sizeof(double) * 3;
  if (sdim == 0)
    qsort(bodies, nbodies, size3, comp_bodies_s0);
  else if (sdim == 1)
    qsort(bodies, nbodies, size3, comp_bodies_s1);
  else if (sdim == 2)
    qsort(bodies, nbodies, size3, comp_bodies_s2);
}

void read_sorted_bodies(int64_t* nbodies, int64_t lbuckets, double* bodies, int64_t buckets[], const char* fname) {
  FILE* file = fopen(fname, "r");
  if (file != NULL) {
    int64_t curr = 1, cbegin = 0, iter = 0, len = *nbodies;
    int ret = !EOF;
    while (iter < len && ret != EOF) {
      int64_t b = 0;
      double x = 0., y = 0., z = 0.;
      ret = fscanf(file, "%lf %lf %lf %" PRId64 "\n", &x, &y, &z, &b);
      if (lbuckets < b)
        len = iter;
      else if (ret != EOF) {
        bodies[iter * 3] = x;
        bodies[iter * 3 + 1] = y;
        bodies[iter * 3 + 2] = z;
        while (curr < b && curr <= lbuckets) {
          buckets[curr - 1] = iter - cbegin;
          cbegin = iter;
          curr++;
        }
        iter++;
      }
    }
    while (curr <= lbuckets) {
      buckets[curr - 1] = iter - cbegin;
      cbegin = iter;
      curr++;
    }
    *nbodies = iter;
    fclose(file);
  }
}

void mat_vec_reference(void(*ef)(double*), int64_t begin, int64_t end, double B[], int64_t nbodies, const double* bodies, const double Xbodies[]) {
  int64_t m = end - begin;
  int64_t n = nbodies;
  for (int64_t i = 0; i < m; i++) {
    int64_t y = begin + i;
    const double* bi = &bodies[y * 3];
    double s = 0.;
    for (int64_t j = 0; j < n; j++) {
      const double* bj = &bodies[j * 3];
      double dX = bi[0] - bj[0];
      double dY = bi[1] - bj[1];
      double dZ = bi[2] - bj[2];
      double r2 = dX * dX + dY * dY + dZ * dZ;
      ef(&r2);
      s = s + r2 * Xbodies[j];
    }
    B[i] = s;
  }
}
