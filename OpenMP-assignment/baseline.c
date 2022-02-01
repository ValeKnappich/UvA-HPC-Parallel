#include <stdlib.h>
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
    approach = baseline; 
    return runBenchmark(multiply);
}