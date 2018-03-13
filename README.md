Parallel Matrix Multiplication using MPI with C++.
======
A simple parallel matrix multiplication code using MPI.

Details
------
Matrices are allocated dynamically and are contiguous in memory.
Matrix sizes don't need to be divisible by the number or processors; the first worker(slave) processor takes care of this.

More information
------
How to compile and run MS MPI programs (using MSVC): [here.](https://blogs.technet.microsoft.com/windowshpc/2015/02/02/how-to-compile-and-run-a-simple-ms-mpi-program/)

About contiguous, dynamic multidimensional array allocation in C++: [here](https://niallpjackson.wordpress.com/2013/06/21/contiguous-dynamic-multidimensional-arrays-in-c/)


*This is my first GitHub commit and was made for learning purposes.*
