//This file contains the first round of optimizations to the tiled 
//matrix multiplication method using AVX SIMD.


#include <iostream>
#include <vector>
#include <immintrin.h>
#include <chrono>
#include "matmul_utils.h"
using namespace std;
using namespace std::chrono;

namespace ops{
    //v2
    //Block Matrix Product
    //Method computes the block matrix product of two 8x8 matrices 
    //To be the cornerstore of the larger matrix product method
    void block_matmul(float* A_ptr, float* B_ptr, float* C_ptr, size_t strideA, size_t strideB_C){
        //Initialize accumulators
        __m256 c0 = _mm256_loadu_ps(C_ptr + 0*strideB_C);
        __m256 c1 = _mm256_loadu_ps(C_ptr + 1*strideB_C);
        __m256 c2 = _mm256_loadu_ps(C_ptr + 2*strideB_C);
        __m256 c3 = _mm256_loadu_ps(C_ptr + 3*strideB_C);
        __m256 c4 = _mm256_loadu_ps(C_ptr + 4*strideB_C);
        __m256 c5 = _mm256_loadu_ps(C_ptr + 5*strideB_C);
        __m256 c6 = _mm256_loadu_ps(C_ptr + 6*strideB_C);
        __m256 c7 = _mm256_loadu_ps(C_ptr + 7*strideB_C);

        for (size_t k = 0; k < 8; k++){
            //Broadcast column of A_block_ptr
            __m256 a0 = _mm256_set1_ps(*(A_ptr + 0*strideA + k));
            __m256 a1 = _mm256_set1_ps(*(A_ptr + 1*strideA + k));
            __m256 a2 = _mm256_set1_ps(*(A_ptr + 2*strideA + k));
            __m256 a3 = _mm256_set1_ps(*(A_ptr + 3*strideA + k));
            __m256 a4 = _mm256_set1_ps(*(A_ptr + 4*strideA + k));
            __m256 a5 = _mm256_set1_ps(*(A_ptr + 5*strideA + k));
            __m256 a6 = _mm256_set1_ps(*(A_ptr + 6*strideA + k));
            __m256 a7 = _mm256_set1_ps(*(A_ptr + 7*strideA + k));

            //Load in row of B_block
            __m256 b0 = _mm256_loadu_ps(B_ptr + strideB_C*k);

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
        _mm256_storeu_ps(C_ptr + 0*strideB_C, c0);
        _mm256_storeu_ps(C_ptr + 1*strideB_C, c1);
        _mm256_storeu_ps(C_ptr + 2*strideB_C, c2);
        _mm256_storeu_ps(C_ptr + 3*strideB_C, c3);
        _mm256_storeu_ps(C_ptr + 4*strideB_C, c4);
        _mm256_storeu_ps(C_ptr + 5*strideB_C, c5);
        _mm256_storeu_ps(C_ptr + 6*strideB_C, c6);
        _mm256_storeu_ps(C_ptr + 7*strideB_C, c7);

    }


    //v2
    //Tiled matrix multiplication
    //IJK loop
    //Will fail on matrices who arent sized correctly
    vector<float> matmul_SIMD_nopad(vector<float>& A, int rowA, int colA, vector<float>& B, int rowB, int colB){
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

                    block_matmul(A_block_ptr, B_block_ptr, C_ptr, colA, colB);

                }
            }
        }

        return C;
    }

    //v2
    //Tiled matrix multiplication
    //IJK loop
    //Padding inside
    vector<float> matmul_SIMD(vector<float>& A, int rowA, int colA, vector<float>& B, int rowB, int colB){
        if (colA != rowB){
            throw invalid_argument("Matrix A dim 1 does not equal matrix B dim 0");
        }

        //Pad matrices with zeros so dimensions are multiples of 8
        vector<int> A_pad = pad(A, rowA, colA);
        vector<int> B_pad = pad(B, rowB, colB);
        
        int rowA_pad = A_pad[0];
        int colA_pad = A_pad[1];
        int rowB_pad = B_pad[0];
        int colB_pad = B_pad[1];
        //Make output vector
        vector<float> C((rowA_pad)*(colB_pad), 0.0f); 

        //Temporarily going to assume matrices are the right size
        for (int i = 0; i < rowA_pad; i+=8){
            for (int j = 0; j < rowB_pad; j+=8){

                    //Make copy of A_block_ptr to feed into block_matmul
                    float* A_block_ptr = A.data() + i*colA_pad + j;
                    
                for (int k = 0; k < colB_pad; k+=8){

                    //Make a copy of B_block to feed into block_matmul
                    float* B_block_ptr = B.data() + j*colB_pad + k;
                    float* C_ptr = C.data() + i*colB_pad + k;

                    block_matmul(A_block_ptr, B_block_ptr, C_ptr, colA_pad, colB_pad);

                }
            }
        }


        //Unpads C so that result has correct size
        unpad(C, rowA, colB, colB_pad-colB);

        return C;
    }


}



//Takes in one of the above matmul functions and runs and records a complete set of experiments
template <typename Func>
void run_experiment(Func f, string algo_name){
    cout << "Running experiment using " << algo_name << endl;
    vector<int> sizes = {16,32,64,128,256,384,512,640,768,896,1024,1280,1536,1792,2048};
    vector<int> times;
    for (int size : sizes){
        vector<float> A(size*size, 0.0f);

        for (int i = 0; i < size*size; i++){
            A[i] = i;
        }

        vector<float> B = A;

        auto start = high_resolution_clock::now();
        vector<float> C = f(A, size, size, B, size, size);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);

        times.push_back(duration.count());

    }

    for (int time : times){
        cout << time << " ";
    }

    record_experiment(algo_name, times);
}


int main(){
    
    run_experiment(ops::matmul_SIMD, "matmul_SIMD_tiled_v2");

    return 0;
}