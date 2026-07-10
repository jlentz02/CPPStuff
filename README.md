# CPPStuff
Learning Summary
This project began as a way for me to learn C++. As a math student, I have long been familiar with python and implemented
some fairly robust projects using Pytorch to handle large unsupervised learning problems (see my repo WDL). 
However, one annoyance I am sure every researcher can sympathize with is experiments and simulations taking a long time to run.
I opted to tackle this problem by learning C++ since C++ is *fast*. Although, as I have explored with this project, there
is a major difference between just writing code in C++ and writing actually performant C++ code.

I believe there is great learning value in executing simple things well, and there is nothing simpler than matrix multiplication.
This simple algorithm became my focus of optimization. Just how far could we go with this simple concept? I am going to recount
my learning by stepping through each algorithm and what I learned by implementing it. You will see me refer to GFLOPs as a
measure of how performant the algorithm is. I do acknowledge that there are better ways to determine algorithm efficiency by
looking at cache misses, but this method is straight forward and easy to extract using the std::chrono library.

- Naive matrix multiplication
    This 





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
- MATMUL v1 - straight forward implementation of tiled matrix multiplication using 
- MATMUL v2 - modification of above to remove copying and use pointers to handle data loading






