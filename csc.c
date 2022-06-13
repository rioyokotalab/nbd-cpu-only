
#include "csc.h"

#include "stdio.h"
#include "stdlib.h"
#include "string.h"

int comp_int64(const void *a, const void *b) {
  int64_t val_a = *(int64_t*)a;
  int64_t val_b = *(int64_t*)b;
  return (int)(val_a - val_b);
}

void cooRemoveDups(int64_t coo[], int64_t* nnz) {
  int64_t len = *nnz;
  if (len > 0) {
    qsort(coo, len, sizeof(int64_t), comp_int64);
    int64_t len_out = 1;
    for (int64_t i = len_out; i < len; i++)
      if (coo[len_out - 1] < coo[i])
        coo[len_out++] = coo[i];
    *nnz = len_out;
  }
}

void cscCreate(struct ColumnSparse* csc, int64_t m, int64_t n, int64_t nnz, const int64_t coo_sorted[]) {
  csc->M = m;
  csc->N = n;
  csc->COL_INDEX = (int64_t*)malloc(sizeof(int64_t) * (n + 1));
  csc->ROW_INDEX = (int64_t*)malloc(sizeof(int64_t) * nnz);
  memset(csc->COL_INDEX, 0, sizeof(int64_t) * (n + 1));
  
  for (int64_t i = 0; i < nnz; i++) {
    int64_t z = coo_sorted[i];
    int64_t x = z / m;
    int64_t y = z - x * m;
    int64_t count_c = csc->COL_INDEX[x];
    csc->COL_INDEX[x] = count_c + 1;
    csc->ROW_INDEX[i] = y;
  }

  int64_t count = 0;
  for (int64_t i = 0; i <= n; i++) {
    int64_t count_c = csc->COL_INDEX[i];
    csc->COL_INDEX[i] = count;
    count = count + count_c;
  }
}

void cscDestroy(struct ColumnSparse* csc) {
  free(csc->COL_INDEX);
  free(csc->ROW_INDEX);
  csc->M = 0;
  csc->N = 0;
  csc->COL_INDEX = NULL;
  csc->ROW_INDEX = NULL;
}
