//Contains padding and unpadding function

#ifndef MATMUL_UTILS_H
#define MATMUL_UTILS_H

#include <vector>
#include <string>

std::vector<int> pad(std::vector<float>& matrix, int rowA, int colA);

void pad(std::vector<float>& matrix, int rowA, int colA, int rowA_pad, int colA_pad);

void unpad(std::vector<float>& matrix, int rowA, int colA, int rowPad);

void print_matrix(std::vector<float>& matrix, int row, int col);

void record_experiment(std::string algo_name, std::vector<int> times);

#endif