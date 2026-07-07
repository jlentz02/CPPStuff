//This file contains my original functions for getting familiar with AVX SIMD instructions
//as well as redudant copies of the methods in MATMULv1.cpp

#include <iostream>
#include <vector>
#include <immintrin.h>
#include <chrono>
using namespace std;
using namespace std::chrono;

namespace vops{
    //SIMD Vector addition for 256 bit i.e. one vector of floats with length 8
    vector<float> add_size8(vector<float>& a, vector<float>& b){
        vector<float> c(a.size());

        __m256 x = _mm256_loadu_ps(a.data());
        __m256 y = _mm256_loadu_ps(b.data());

        __m256 z = _mm256_add_ps(x,y);

        _mm256_storeu_ps(c.data(), z);
        
        return c;
    }

    //SIMD vector addition for arbitrarily sized vectors
    vector<float> add(vector<float>& a, vector<float>& b){
        if (a.size() != b.size()){
            throw invalid_argument("Vector sizes do not align");
        }

        vector<float> c(a.size());

        size_t i = 0;
        for (; i + 7 < a.size(); i += 8){
            __m256 x = _mm256_loadu_ps(a.data() + i);
            __m256 y = _mm256_loadu_ps(b.data() + i);

            __m256 z = _mm256_add_ps(x,y);

            _mm256_storeu_ps(c.data() + i, z);
        }

        for (; i < a.size(); i++){
            c[i] = a[i] + b[i];
        }

        return c;

    }

    //SIMD dot product with horizontal addition
    float dot(vector<float>& a, vector<float>& b){
        if (a.size() != b.size()){
            throw invalid_argument("Vector sizes do not align");
        } 
        
        float c = 0.0f;

        __m256 z = _mm256_setzero_ps();

        size_t i = 0;
        for (; i + 7 < a.size(); i += 8){
            __m256 x = _mm256_loadu_ps(a.data() + i);
            __m256 y = _mm256_loadu_ps(b.data() + i);

            z = _mm256_fmadd_ps(x, y, z);

        }

        __m128 hi = _mm256_extractf128_ps(z, 1);
        __m128 lo = _mm256_castps256_ps128(z);

        lo = _mm_add_ps(lo, hi);

        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);

        c += _mm_cvtss_f32(lo);

        for (; i < a.size(); i++){
            c += a[i]*b[i];
        }

        return c;
    }

    //SIMD dot product with horizontal addition
    //Modified to accept pointers for use in other function
    float dot_ptr(float* a, int a_size, float* b, int b_size){
        if (a_size != b_size){
            throw invalid_argument("Vector sizes do not align");
        } 
        
        float c = 0.0f;

        __m256 z = _mm256_setzero_ps();

        size_t i = 0;
        for (; i + 7 < a_size; i += 8){
            __m256 x = _mm256_loadu_ps(a + i);
            __m256 y = _mm256_loadu_ps(b + i);

            z = _mm256_fmadd_ps(x, y, z);

        }

        __m128 hi = _mm256_extractf128_ps(z, 1);
        __m128 lo = _mm256_castps256_ps128(z);

        lo = _mm_add_ps(lo, hi);

        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);

        c += _mm_cvtss_f32(lo);

        for (; i < a_size; i++){
            c += a[i]*b[i];
        }

        return c;
    }

    //Matrix Vector Product
    //We used a flattened matrix
    //I am going to leverage the dot-product function from above
    vector<float> mat_vec(vector<float>& A, int rowA, int colA, vector<float>&x){
        if (colA != x.size()){
            throw invalid_argument("Matrix dim 1 does not equal vector length");
        }

        vector<float> b(rowA, 0.0f);
        float* x_ptr = x.data();
        for (size_t i = 0; i < rowA; i++){
            float* row_ptr = A.data() + i*colA;

            b[i] = dot_ptr(row_ptr, colA, x_ptr, x.size());

        }
        return b;
    }

    //Matrix Matrix Product 
    //Method uses flattened matrices and presupposes the second matrix has already been transposed
    //Method leverages the dot product function above
    //As such, this could be considered a naive SIMD implementation and should be later bested by a tiled version
    vector<float> matmul_SIMD_naive(vector<float>& A, int rowA, int colA, vector<float>& B, int rowB, int colB){
        if (colA != rowB){
            throw invalid_argument("Matrix A dim 1 does not equal matrix B dim 0");
        }

        vector<float> C(rowA*colB, 0.0f);
        for (size_t i = 0; i < rowA; i++){
            for (size_t k = 0; k < colB; k++){
                C[i*colB + k] = dot_ptr(A.data() + i*colA, colA, B.data() + k*rowB, rowB);
            }
        }

        return C;
    }

    //v1
    //Block Matrix Product
    //Method computes the block matrix product of two 8x8 matrices 
    //To be the cornerstore of the larger matrix product method
    void block_matmul(float* A_block, float* B_block, float* C_ptr, size_t N){
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
            //Broadcast column of A_block
            __m256 a0 = _mm256_set1_ps(*(A_block + 0*8 + k));
            __m256 a1 = _mm256_set1_ps(*(A_block + 1*8 + k));
            __m256 a2 = _mm256_set1_ps(*(A_block + 2*8 + k));
            __m256 a3 = _mm256_set1_ps(*(A_block + 3*8 + k));
            __m256 a4 = _mm256_set1_ps(*(A_block + 4*8 + k));
            __m256 a5 = _mm256_set1_ps(*(A_block + 5*8 + k));
            __m256 a6 = _mm256_set1_ps(*(A_block + 6*8 + k));
            __m256 a7 = _mm256_set1_ps(*(A_block + 7*8 + k));

            //Load in row of B_block
            __m256 b0 = _mm256_loadu_ps(B_block + 8*k);

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
    //Block Matrix Product
    //Method deviates from v1 by using aligned load for A as I am modifying A_block to be aligned
    //Method also loads rows of B_block similarly to how C_block is loaded eliminating a wasteful
    //copying step in the main matmul method.
    void block_matmul_v2(float* A_block, float* B_ptr, float* C_ptr, size_t N){
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
            //Broadcast column of A_block
            __m256 a0 = _mm256_set1_ps(*(A_block + 0*8 + k));
            __m256 a1 = _mm256_set1_ps(*(A_block + 1*8 + k));
            __m256 a2 = _mm256_set1_ps(*(A_block + 2*8 + k));
            __m256 a3 = _mm256_set1_ps(*(A_block + 3*8 + k));
            __m256 a4 = _mm256_set1_ps(*(A_block + 4*8 + k));
            __m256 a5 = _mm256_set1_ps(*(A_block + 5*8 + k));
            __m256 a6 = _mm256_set1_ps(*(A_block + 6*8 + k));
            __m256 a7 = _mm256_set1_ps(*(A_block + 7*8 + k));

            //Load in row of B_block
            __m256 b0 = _mm256_loadu_ps(B_ptr + k*N);

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

    //v1
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
                //Make copy of A_block to feed into block_matmul
                    float A_block[64] = {0.0f};
                    for (int ii = 0; ii < 8; ii++){
                        for (int jj = 0; jj < 8; jj++){
                            A_block[ii*8 + jj] = A[(ii + i)*colA + jj + j];
                        }
                    }
   
                for (int k = 0; k + 7 < colB; k+=8){

                    //Make a copy of B_block to feed into block_matmul
                    float B_block[64] = {0.0f};
                    for (int jj = 0; jj < 8; jj++){
                        for (int kk = 0; kk < 8; kk++){
                            B_block[jj*8 + kk] = B[(jj + j)*colB + kk + k];
                        }
                    }
                    float* C_ptr = C.data() + i*colB + k;
                    block_matmul(A_block, B_block, C_ptr, colB);

                }
            }
        }

        return C;
    }

    //v2
    //Tiled matrix multiplication
    //IKJ loop, actually less efficient since we need to make a copy for every i, j, and k. The above only copies
    //A_block for each i and j
    vector<float> matmul_SIMD_IKJ(vector<float>& A, int rowA, int colA, vector<float>& B, int rowB, int colB){
        if (colA != rowB){
            throw invalid_argument("Matrix A dim 1 does not equal matrix B dim 0");
        }

        //Make output vector
        vector<float> C(rowA*colB, 0.0f); 

        //Temporarily going to assume matrices are the right size
        for (int i = 0; i+7 < rowA; i+=8){
            for (int k = 0; k + 7 < colB; k+=8){

                float* C_ptr = C.data() + i*colB + k;

                for (int j = 0; j + 7 < rowB; j+=8){

                    //Make copy of A_block to feed into block_matmul
                    float A_block[64] = {0.0f};
                    for (int ii = 0; ii < 8; ii++){
                        for (int jj = 0; jj < 8; jj++){
                            A_block[ii*8 + jj] = A[(ii + i)*colA + jj + j];
                        }
                    }

                    //Make a copy of B_block to feed into block_matmul
                    float B_block[64] = {0.0f};
                    for (int jj = 0; jj < 8; jj++){
                        for (int kk = 0; kk < 8; kk++){
                            B_block[jj*8 + kk] = B[(jj + j)*colB + kk + k];
                        }
                    }

                    block_matmul(A_block, B_block, C_ptr, colB);

                }
   

            }
        }

        return C;
    }


    //v3
    //Tiled matrix multiplication
    //IJK loop
    //Doesn't load B_block and does it inside the tiled matmul instead
    vector<float> matmul_SIMD_v3(vector<float>& A, int rowA, int colA, vector<float>& B, int rowB, int colB){
        if (colA != rowB){
            throw invalid_argument("Matrix A dim 1 does not equal matrix B dim 0");
        }

        //Make output vector
        vector<float> C(rowA*colB, 0.0f); 

        //Temporarily going to assume matrices are the right size
        for (int i = 0; i+7 < rowA; i+=8){
            for (int j = 0; j + 7 < rowB; j+=8){
                //Make copy of A_block to feed into block_matmul
                    float A_block[64] = {0.0f};
                    for (int ii = 0; ii < 8; ii++){
                        for (int jj = 0; jj < 8; jj++){
                            A_block[ii*8 + jj] = A[(ii + i)*colA + jj + j];
                        }
                    }
   
                for (int k = 0; k + 7 < colB; k+=8){

                    //Make a copy of B_block to feed into block_matmul
                    float B_block[64] = {0.0f};
                    for (int jj = 0; jj < 8; jj++){
                        for (int kk = 0; kk < 8; kk++){
                            B_block[jj*8 + kk] = B[(jj + j)*colB + kk + k];
                        }
                    }
                    float* C_ptr = C.data() + i*colB + k;
                    block_matmul(A_block, B_block, C_ptr, colB);

                }
            }
        }

        return C;
    }

}



int main(){
    int size = 1000;
    vector<float> A(size*size, 0.0f);
    vector<float> B(size*size, 0.0f);

    for (int i = 0; i < size*size; i++){
        A[i] = i;
    }

    B = A;

    auto start = high_resolution_clock::now();
    vector<float> C = vops::matmul_SIMD(A, size, size, B, size, size);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    cout << duration.count() << endl;

    return 0;
}