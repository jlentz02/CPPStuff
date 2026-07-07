//This file contains the first round of optimizations to the tiled 
//matrix multiplication method using AVX SIMD.


#include <iostream>
#include <vector>
#include <immintrin.h>
#include <chrono>
#include "matmul_utils.h"
using namespace std;
using namespace std::chrono;

namespace vops{
    //v2
    //Block Matrix Product
    //Method computes the block matrix product of two 8x8 matrices 
    //To be the cornerstore of the larger matrix product method
    void block_matmul(float* A_ptr, float* B_ptr, float* C_ptr, size_t N){
        //Initialize accumulators
        __m256 c0 = _mm256_loadu_ps(C_ptr + 0*N);
        __m256 c1 = _mm256_loadu_ps(C_ptr + 1*N);
        __m256 c2 = _mm256_loadu_ps(C_ptr + 2*N);
        __m256 c3 = _mm256_loadu_ps(C_ptr + 3*N);
        __m256 c4 = _mm256_loadu_ps(C_ptr + 4*N);
        __m256 c5 = _mm256_loadu_ps(C_ptr + 5*N);
        __m256 c6 = _mm256_loadu_ps(C_ptr + 6*N);
        __m256 c7 = _mm256_loadu_ps(C_ptr + 7*N);

        for (size_t k = 0; k < 8; k++){
            //Broadcast column of A_block_ptr
            __m256 a0 = _mm256_set1_ps(*(A_ptr + 0*N + k));
            __m256 a1 = _mm256_set1_ps(*(A_ptr + 1*N + k));
            __m256 a2 = _mm256_set1_ps(*(A_ptr + 2*N + k));
            __m256 a3 = _mm256_set1_ps(*(A_ptr + 3*N + k));
            __m256 a4 = _mm256_set1_ps(*(A_ptr + 4*N + k));
            __m256 a5 = _mm256_set1_ps(*(A_ptr + 5*N + k));
            __m256 a6 = _mm256_set1_ps(*(A_ptr + 6*N + k));
            __m256 a7 = _mm256_set1_ps(*(A_ptr + 7*N + k));

            //Load in row of B_block
            __m256 b0 = _mm256_loadu_ps(B_ptr + N*k);

            //FMA step
            c0 = _mm256_fmadd_ps(a0, b0, c0);
            c1 = _mm256_fmadd_ps(a1, b0, c1);
            c2 = _mm256_fmadd_ps(a2, b0, c2);
            c3 = _mm256_fmadd_ps(a3, b0, c3);
            c4 = _mm256_fmadd_ps(a4, b0, c4);
            c5 = _mm256_fmadd_ps(a5, b0, c5);
            c6 = _mm256_fmadd_ps(a6, b0, c6);
            c7 = _mm256_fmadd_ps(a7, b0, c7);
        }

        //Store step
        _mm256_storeu_ps(C_ptr + 0*N, c0);
        _mm256_storeu_ps(C_ptr + 1*N, c1);
        _mm256_storeu_ps(C_ptr + 2*N, c2);
        _mm256_storeu_ps(C_ptr + 3*N, c3);
        _mm256_storeu_ps(C_ptr + 4*N, c4);
        _mm256_storeu_ps(C_ptr + 5*N, c5);
        _mm256_storeu_ps(C_ptr + 6*N, c6);
        _mm256_storeu_ps(C_ptr + 7*N, c7);

    }


    //v2
    //Tiled matrix multiplication
    //IJK loop
    vector<float> matmul_SIMD(vector<float>& A, int rowA, int colA, vector<float>& B, int rowB, int colB){
        if (colA != rowB){
            throw invalid_argument("Matrix A dim 1 does not equal matrix B dim 0");
        }

        //Make output vector
        vector<float> C(rowA*colB, 0.0f); 

        //Temporarily going to assume matrices are the right size
        for (int i = 0; i+7 < rowA; i+=8){
            for (int j = 0; j + 7 < rowB; j+=8){

                    //Make copy of A_block_ptr to feed into block_matmul
                    float* A_block_ptr = A.data() + i*colA + j;
                    
                for (int k = 0; k + 7 < colB; k+=8){

                    //Make a copy of B_block to feed into block_matmul
                    float* B_block_ptr = B.data() + j*colB + k;
                    float* C_ptr = C.data() + i*colB + k;

                    block_matmul(A_block_ptr, B_block_ptr, C_ptr, colB);

                }
            }
        }

        return C;
    }


}



int main(){
    int size = 22;
    vector<float> A(size*size, 0.0f);
    vector<float> B(size*size, 0.0f);

    for (int i = 0; i < size*size; i++){
        A[i] = 1;
    }

    B = A;

    vector<float> A_padded = pad(A, size, size);

    for (int i = 0; i < 24; i++){
        for (int j = 0; j < 24; j++){
            cout << A_padded[24*i + j] << " ";
        }
        cout << endl;
    }

    return 0;
}