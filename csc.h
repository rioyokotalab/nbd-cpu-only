
#pragma once

#include "stddef.h"
#include "stdint.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ColumnSparse {
  int64_t M;
  int64_t N;
  int64_t* COL_INDEX;
  int64_t* ROW_INDEX;
};

void cooRemoveDups(int64_t coo[], int64_t* nnz);

void cscCreate(struct ColumnSparse* csc, int64_t m, int64_t n, int64_t nnz, const int64_t coo_sorted[]);

void cscDestroy(struct ColumnSparse* csc);


#ifdef __cplusplus
}
#endif
