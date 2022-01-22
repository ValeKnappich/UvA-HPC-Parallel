#include <stdio.h>
#include <stdlib.h>


/**
 * Construct vector with the numbers from 1 to N
 * @param N Number of elements in the returned vectors
*/
double* getVector(int N) {
    double* vector = (double*) malloc(N * sizeof(double));
    for (int i = 0; i < N; i++){
        vector[i] = i + 1;
    }
    return vector;
}


/**
 * Construct an Identity matrix as flattened array.
 * In memory, the rows are simply concatenated.
 * @param N dimension of resulting matrix (NxN) 
*/
// double* getIdentity(int N) {
//     double* identity = (double*) malloc(N * N * sizeof(double)); // Allocate memory
//     if(identity == NULL){
//         printf("Allocating %ld Bytes failed. Aborting.", N * N * sizeof(double));
//         exit(1);
//     }
//     for(int i = 0; i < N; i++){
//         for(int j = 0; j < N; j++){
//             if (i == j) 
//                 identity[i * N + j] = 1;    // Diagonal values
//             else 
//                 identity[i * N + j] = 0;    // All other values
//         }
//     }
//     return identity;
// }

double** getIdentity(int N) {
    // Allocate memory
    double** identity = (double**) malloc(N * sizeof(double*)); // Allocate memory for pointers to rows
    if(identity == NULL){
        printf("Allocating %ld Bytes failed. Aborting.", N * N * sizeof(double));
        exit(1);
    }
    for(int row = 0; row < N; row++){
        identity[row] = malloc(N * sizeof(double));   // Allocate memory per row
        if(identity == NULL){
            printf("Allocating %ld Bytes failed. Aborting.", N * sizeof(double));
            exit(1);
        }
    }

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            if (i == j) 
                identity[i][j] = 1;    // Diagonal values
            else 
                identity[i][j] = 0;    // All other values
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
void printMatrix(double** matrix, int N, int M){
    printf("[");
    for (int i = 0; i < N; i++){
        if (i != 0) printf(" "); // offset of 1 for every row but the first
        printf("[");
        for (int j = 0; j < M; j++){
            printf("%2.2f", matrix[i][j]);
            if (j != M - 1) printf(" "); // add space separator unless after last value of row
        }
        printf("]");
        if (i == N - 1) printf("]");
        else printf("\n");
    }
    printf("\n");
}

void printVector(double* vector, int N){
    printf("[");
    for (int i = 0; i < N; i++){
        printf("%2.2f", vector[i]);
        if (i != N - 1) printf(" "); // add space separator unless after last value of row
    }
    printf("]\n");
}