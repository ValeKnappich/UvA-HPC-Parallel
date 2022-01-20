#include <stdio.h>
#include <stdlib.h>


/* Construct vector with the numbers from 1 to N */
double* getVector(int N) {
    double* vector = (double*) malloc(N * sizeof(double));
    for (int i = 0; i < N; i++){
        vector[i] = i + 1;
    }
    return vector;
}


/* Construct an Identity matrix */
double* getIdentity(int N) {
    double* identity = (double*) malloc(N * N * sizeof(double)); // Allocate memory
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            if (i == j) 
                identity[i * N + j] = 1;    // Diagonal values
            else 
                identity[i * N + j] = 0;    // All other values
        }
    }
    return identity;
}


/**
 * Print a formatted Matrix
 * 
 * Example Result from 5-identity
 * [[1 0 0 0 0]
 *  [0 1 0 0 0]
 *  [0 0 1 0 0]
 *  [0 0 0 1 0]
 *  [0 0 0 0 1]]
 * @param matrix Pointer to first entry of the matrix
 * @param N Number of rows of the matrix
 * @param M Number of columns of the matrix

*/
void printMatrix(double* matrix, int N, int M){
    printf("[");
    for (int i = 0; i < N; i++){
        if (i != 0) printf(" "); // offset of 1 for every row but the first
        printf("[");
        for (int j = 0; j < M; j++){
            printf("%2.2f", matrix[i * M + j]);
            if (j != M - 1) printf(" "); // add space separator unless after last value of row
        }
        printf("]");
        if (i == N - 1) printf("]");
        else printf("\n");
    }
    printf("\n");
}