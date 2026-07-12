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
    - This is the most basic, and straightforward, implementation of matrix multiplication that I wrote. Prior to writing
      this method, I became familiar with loops, vectors, classes, and the basic syntax of C++. For this method in particular,
      I treated a matrix as a class object whos main data was stored in a vector of vectors. This is bad for a number of reasons
      but it is naive matrix multiplication for a reason. Anyway, this method clocked in at ~0.5 GFLOPs on a 2048x2048 matrix multiplication.
      All references to GFLOPs going forward correspond to a 2048x2048 matrix multiplication.
- Naive matrix multiplication transposed
    - If we transpose the second matrix we can do operations row by row instead of row by column drastically reducing the number of
      memory switches. Unfortunately, this had no effect (possibly due to poor compiler optimizations but we will revisit that at the end).
      This implementation performed marginally worse than the prior implementation, also around ~0.5 GFLOPs.
- Vectorized matrix multiplication
    - Flattening your matrix makes it so that all of the data is stored sequentially in memory. This offered a huge improvement getting us
      to ~0.8 GFLOPs.
- Vectorized matrix multiplication transposed
    - Same idea as the previous transposed function, same result with little change. ~0.8 GFLOPs.
- SIMD dot product matrix multiplication
    - Now we are getting to the good stuff. Matrix multiplication is a easily parallelizable operation (thats why GPUs are so good at it).
      However, up until this point we have been executing every single addition and multiplication one at a time, all ~8 billion of them.
      We can do much better by leveraging our CPU to compute math on 32 byte blocks of data (8 floats) at the same time using AVX SIMD
      instructions. These just tell the CPU to execute the same operation in parallel on 8 floats at a time. This basic implementation
      computes each dot product between a row of matrix A and a column of matrix B to produce the resultant entry of matrix C.
      Even this naive approach leveraging AVX intrinsics yielded ~4 GFLOPs.
- SIMD tiled version 1
    - This is the bread and butter of high performance matrix multiplication. If your CPU can leverage SIMD to compute very quickly
      on 8 float wide pieces of information, then we should design the entire algorithm around this. Math students will be familiar
      with block matrix multiplication, and that is essentially the principle we are using here except it is called tiling. My method
      would iterate through the matrix using three nested loops with a stride of 8, and then compute the relevant 8x8 block matrix
      multiplications. However, as I originally wrote my block matrix multiplication code, the inputs had to be 8x8 block matrices themselves.
      So, I wound up with this very wasteful copying step that really ate into performance in hindsight. This method reached ~5.7 GFLOPs.
    - It was around this point in the process that the project had grown sufficiently that I wanted to write some shared method between
      files. This is where the utilites library I wrote came into play, and when I first started using CMake to handle compiling the
      separate files in the library. CMake sucked at first, but I think it is fairly intuitive once you get it going and it made handling
      shared methods much easier. In particular, I needed a method to pad odd sized matrices with zeros so their dimensions were
      multiples of 8.
- SIMD tiled version 2
    - This is, at current, my best implementation of matrix multiplication. It discards the copying step entirely and handles all of the
      data loading using pointers. I would say this is the point in my learning C++ where pointers started to really make sense, and where
      I really started to appreciate just how useful they were. This method clocked in at ~11 GFLOPs.
    - 





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






