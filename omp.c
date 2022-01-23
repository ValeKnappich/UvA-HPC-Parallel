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
double* multiply(double** matrix, double* vector, const long N){
    double* result = (double*) malloc(N * sizeof(double));
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
    runSanityCheck(multiply);
    approach = omp;
    return runBenchmark(multiply);    
}

