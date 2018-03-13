// Simple matrix multiplication using MPI
//
// Solves a*b = c
//
// Matrix Sizes:	a[NROWSA][NCOLSA]
//					b[NCOLSA][NCOLSB]
//					c[NROWSA][NCOLSB]
//
// Matrices allocated contiguously in memory

#include "mpi.h"
#include <chrono>
#include <iostream>


int main(int argc, char **argv)
{
	int myrank = 0, numprocs = 0, numworkers = 0, offset = 0, rows = 0 ;
	int k = 1, dest = 0, source = 0;
	int tag1 = 1, tag2 = 2;

	//contiguous dynamic allocation for matrices a, b, c
	int NROWSA = 2;
	int NCOLSA = 3;
	int NCOLSB = 2;
	
	int **a = new int*[NROWSA];
	int **b = new int*[NCOLSA];
	int **c = new int*[NROWSA];

	int sizeA = NROWSA*NCOLSA;
	int sizeB = NCOLSA*NCOLSB;
	int sizeC = NROWSA*NCOLSB;

	a[0] = new int[sizeA];
	b[0] = new int[sizeB];
	c[0] = new int[sizeC];

	for (int i = 1; i < NROWSA; i++)
	{
		a[i] = &a[0][i*NCOLSA];
		c[i] = &c[0][i*NCOLSB]; // don't need a separate loop for this, c[NROWSA][NCOLSB]
	}

	for (int i = 1; i < NCOLSA; i++)
		b[i] = &b[0][i*NCOLSB];
	//end allocations

	//turn on MPI
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);	//get number of procs
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);		//assign a myrank for each proc
	
	numworkers = numprocs - 1; //number of worker threads

	//code for master process (rank 0)						
	if (myrank == 0)
	{
		//initialize matrix a
		for (int i = 0; i < NROWSA; i++)
		{
			for (int j = 0; j < NCOLSA; j++)
			{
				a[i][j] = k;
				k++;
			}
		}

		//initialize matrix b
		for (int i = 0; i < NCOLSA; i++)
		{
			for (int j = 0; j < NCOLSB; j++)
			{
				b[i][j] = k;
				k++;
			}
		}

		//initialize matrix c
		for (int i = 0; i < NROWSA; i++)
		{
			//could have included this in the loop for matrix a initalization; kept here for readability
			for (int j = 0; j < NCOLSB; j++)
				c[i][j] = 0;
		}


		int avrows = NROWSA / numworkers;
		int extra = NROWSA % numworkers; // we'll make worker1 process any extra rows 
		auto start = std::chrono::steady_clock::now();

		//send to workers
		for (dest = 1; dest <= numworkers; dest++)
		{	
			// if there are any extra rows, worker1 will take care of them
			rows = (extra != 0) ? avrows + extra : avrows; // if no extra rows, process avrows only

			//send the offset
			MPI_Send(&offset, 1, MPI_INT, dest, tag1, MPI_COMM_WORLD);
			//send the number of rows to be sent
			MPI_Send(&rows, 1, MPI_INT, dest, tag1, MPI_COMM_WORLD);
			//send corresponding parts of matrix a to each worker
			MPI_Send(&(a[offset][0]), rows*NCOLSA, MPI_INT, dest, tag1, MPI_COMM_WORLD);
			//send whole matrix b to each worker
			MPI_Send(&(b[0][0]), NCOLSA*NCOLSB, MPI_INT, dest, tag1, MPI_COMM_WORLD);
			offset = offset + rows;
			extra = 0; // after sending the first batch, there are no more extra rows for the other workers
		}

		//recieve results from workers
		for (int i = 1; i <= numworkers; i++)
		{
			source = i;
			MPI_Recv(&offset, 1, MPI_INT, source, tag2, MPI_COMM_WORLD, &status);
			MPI_Recv(&rows, 1, MPI_INT, source, tag2, MPI_COMM_WORLD, &status);
			MPI_Recv(&c[offset][0], rows*NCOLSB, MPI_INT, source, tag2, MPI_COMM_WORLD, &status);
		}


		//display result - use small matrices for testing purposes
		for (int i = 0; i < NROWSA; i++)
		{
			printf("\n");
			for (int j = 0; j < NCOLSB; j++) {
				printf("%12d", c[i][j]);
			}
		}

		auto end = std::chrono::steady_clock::now();
		auto diff = end - start;
		std::cout << "\n\n" << "Computation Completed in "
					<< std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() << " ms"
					<< "\n";


		// done; get rid of everything
		delete[] a[0];
		delete[] b[0];
		delete[] c[0];
		delete[] a;
		delete[] b;
		delete[] c;
	}
	else //for worker threads (rank > 0)
	{	
		//each worker recieves information from master (with tag 1)
		MPI_Recv(&offset, 1, MPI_INT, 0, tag1, MPI_COMM_WORLD, &status);
		MPI_Recv(&rows, 1, MPI_INT, 0, tag1, MPI_COMM_WORLD, &status);
		MPI_Recv(&(a[offset][0]), rows*NCOLSA, MPI_INT, 0, tag1, MPI_COMM_WORLD, &status);
		MPI_Recv(&(b[0][0]), NCOLSA*NCOLSB, MPI_INT, 0, tag1, MPI_COMM_WORLD, &status);

		//actual computation
		for (int i = 0; i < NROWSA; i++)
		{
			for (int j = 0; j <NCOLSB; j++)
			{
				c[i][j] = 0;
				for (k = 0; k <NCOLSA; k++)
					c[i][j] = c[i][j] + a[i][k] * b[k][j];
			}
		}

		//send back result from each worker
		MPI_Send(&offset, 1, MPI_INT, 0, tag2, MPI_COMM_WORLD);
		MPI_Send(&rows, 1, MPI_INT, 0, tag2, MPI_COMM_WORLD);
		MPI_Send(&(c[offset][0]), rows*NCOLSB, MPI_INT, 0, tag2, MPI_COMM_WORLD);
	}
	
	//terminate MPI
	MPI_Finalize();
	return 0;
}

