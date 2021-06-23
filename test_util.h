
#pragma once

#include <iostream>

void printVec(double* a, int n, int inc) {
  for (int i = 0; i < n; i++)
    std::cout << a[i * inc] << " ";
  std::cout << std::endl;
}

void printMat(double* a, int m, int n, int lda) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++)
      std::cout << a[i + j * lda] << " ";
    std::cout << std::endl;
  }
}