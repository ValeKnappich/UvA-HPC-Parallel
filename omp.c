#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "utils.h"


/**
 * Multiply a square matrix and a vector in parallel using OpenMP.
 * Each process calculates the inner product of the vector and a row of the matrix.
 * @param matrix Pointer to matrix
 * @param vector Pointer to vector
 * @param N dimension of matrix (NxN) and vector (N)
*/
double* multiply(double* matrix, double* vector, int N){
    double* result = (double*) malloc(N * sizeof(double));

    #pragma omp parallel for
    for (int i = 0; i < N; i++){
        double* row = matrix + i * N; // Pointer arithmetic to get row of matrix
        int rowSum = 0;
        for (int j = 0; j < N; j++){
            rowSum += vector[j] * row[j];
        }
        result[i] = rowSum;
    }
    return result;
}


int main() {
    // Sanity Check to see if multiplication works
    double* testMatrix = getIdentity(5);
    double* testVector = getVector(5);
    printf("Sanity Check: \n");
    printf("Matrix: \n");
    printMatrix(testMatrix, 5, 5);
    printf("Vector: \n");
    printMatrix(testVector, 1, 5);
    printf("Product: \n");
    double* prod = multiply(testMatrix, testVector, 5);
    printMatrix(prod, 1, 5);

    // Constants
    const int N = 10000;
    const int nIterations = 1000;

    // Construct Data
    double* identity = getIdentity(N);
    double* vector = getVector(N);

    // Benchmark multiple iterations
    const double start = omp_get_wtime();
    for (int i = 0; i < nIterations; i++) {
        vector = multiply(identity, vector, N);
    }

    const double end = omp_get_wtime();
    printf("Ran %d iterations of 'omp' with rank %d in %.2f seconds using %d threads\n", 
           nIterations, N, end - start, omp_get_max_threads());
}

