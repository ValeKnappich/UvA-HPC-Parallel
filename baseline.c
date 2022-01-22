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
double* multiply(double** matrix, double* vector, const long N, double* result){
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
    time_t start, end;
    start = clock();

    for (int i = 0; i < nIterations; i++) {
        vector = multiply(identity, vector, N, result);
    }

    end = clock();
    printf("Ran %d iterations of 'baseline' with rank %d in %.2f seconds\n", 
           nIterations, N, (end - start)/(double) CLOCKS_PER_SEC);
}