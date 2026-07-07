//Contains padding and unpadding function

#ifndef MATMUL_UTILS_H
#define MATMUL_UTILS_H

#include <vector>

std::vector<float> pad(const std::vector<float>& matrix, int rowA, int colA);

std::vector<float> unpad(const std::vector<float>& matrix, int rowA, int colA);

#endif