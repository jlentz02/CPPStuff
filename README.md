# CPPStuff
C++ stuff
This repo is going to be a collection of small projects and other one-off bits of code to help me learn C++.

## Matrix Libary
My first project idea is to do matrix multiplication as fast as I am able. I want to compare the most naive implementation to increasingly more optimized and memory efficient code.
This should teach me about memory efficiency, SIMD, best practices for storing data, etc. 

Once I get some results from different methods, I will test them on trials of 1000x1000 matrix multiplication and put them in a table here. 

Currently taking a detour to become more familiar with AVX intrisics.

So far I have the following methods:
- Transpose
- Print
- Flatten
- Unflatten
- Reshape
- Naive Matrix Multiplication (vector of vectors)
- Naive Matrix Multiplication with transposed second matrix
- Vectorized Matrix Multiplication
- Vectorized Matrix Multiplication with transposed second matrix*

*Fastest method so far mostly due to the efficient memory access of the vectors.
This method is about 75% faster than the naive implementation. 

## SIMD Stuff
As mentioned above to progress in multipliying matrices faster, the next thing I am looking at is SIMD.
In particular, I am writing code for the AVX operation set, but may also write for NEON later since I am
working across multiple systems. As such, this folder is just a collection of methods as I become 
comfortable with the syntax and work towards matrix by matrix multiplication.

So far I have the following methods implemented with SIMD:
- Vector by Vector Addition
- Dot Product
- Matrix Vector Product (naive using dot product)
- Matrix Matrix Product (naive using dot product)
- Block Matrix Product
- Tiled Matrix Matrix Product (using block matrix product) (MATMULv1)
- MATMUL v2 - modification of above to remove copying and use pointers to handle data load
- MATMUL v3 - modification of above to use aligned load and arrays internally instead of unaligned vectors





