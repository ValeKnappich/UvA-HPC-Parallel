#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "utils.h"


/**
 * Multiply a square matrix and a vector sequentially
 * @param matrix Pointer to matrix
 * @param vector Pointer to vector
 * @param N dimension of matrix (NxN) and vector (N)
*/
double* multiply(double** matrix, double* vector, const long N){
    double* result = (double*) malloc(N * sizeof(double));
    for (int i = 0; i < N; i++){
        double rowSum = 0;
        for (int j = 0; j < N; j++){
            rowSum += vector[j] * matrix[i][j];
        }
        result[i] = rowSum;
    }
    return result;
}


int main() {
    runSanityCheck(multiply);

    // Constants
    const int N = 100000;
    const int nIterations = 10;

    // Construct Data
    double** identity = getIdentity(N);
    double* vector = getVector(N);

    // Benchmark multiple iterations
    time_t start, end;
    start = clock();

    for (int i = 0; i < nIterations; i++) {
        double* newVector = multiply(identity, vector, N);
        free(vector);   // Avoid memory leaks, since a new result vector is allocated in multiply
        vector = newVector;
    }

    end = clock();
    printf("Ran %d iterations of 'baseline' with rank %d in %.2f seconds\n", 
           nIterations, N, (end - start) / (double) CLOCKS_PER_SEC);
}