#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h> // for memcpy
#include "utils.h"


/**
 * Multiply a square matrix and a vector in parallel using OpenMP.
 * Each process calculates the inner product of the vector and a row of the matrix.
 * @param matrix Pointer to matrix
 * @param vector Pointer to vector
 * @param N dimension of matrix (NxN) and vector (N)
*/
double* multiply(double** matrix, double* vector, const long N, double* result){ 
    #pragma omp parallel for
    for (int i = 0; i < N; i++){
        double rowSum = 0;
        #pragma omp parallel for reduction(+:rowSum)
        for (int j = 0; j < N; j++){
            rowSum += vector[j] * matrix[i][j];
        }
        result[i] = rowSum;
    }
    return result;
}


int main() {
    // Sanity Check to see if multiplication works
    double** testMatrix = getIdentity(5);
    double* testVector = getVector(5);
    double* testResult = (double*) malloc(5 * sizeof(double));
    printf("Sanity Check: \n");
    printf("Matrix: \n");
    printMatrix(testMatrix, 5, 5);
    printf("Vector: \n");
    printVector(testVector, 5);
    printf("Product: \n");
    multiply(testMatrix, testVector, 5, testResult);
    printVector(testResult, 5);

    // Constants
    const int N = 100000;
    const int nIterations = 10;

    // Construct Data
    double** identity = getIdentity(N);
    double* vector = getVector(N);
    double* result = (double*) malloc(N * sizeof(double));

    // Benchmark multiple iterations
    const double start = omp_get_wtime();
    for (int i = 0; i < nIterations; i++) {
        vector = multiply(identity, vector, N, result);
    }
    const double end = omp_get_wtime();
    printf("Ran %d iterations of 'omp' with rank %d in %.2f seconds using %d threads\n", 
           nIterations, N, end - start, omp_get_max_threads());
}

