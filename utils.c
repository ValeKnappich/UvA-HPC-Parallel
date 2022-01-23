#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include "utils.h"


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
 * Construct an Identity matrix.
 * Memory is allocated row-wise.
 * @param N dimension of resulting matrix (NxN) 
*/
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


/**
 * Print a formatted vector.
 * E.g. [1.00 2.00 3.00 4.00 5.00]
 * @param vector Vector to print
 * @param N Length of the vector
*/
void printVector(double* vector, int N){
    printf("[");
    for (int i = 0; i < N; i++){
        printf("%2.2f", vector[i]);
        if (i != N - 1) printf(" "); // add space separator unless after last value of row
    }
    printf("]\n");
}


/**
 * Run a sanity check example of multiplying the identity matrix with a test vector.
 * @param multiply Function pointer to function with interface as defined in 'multiply_t'
*/
void runSanityCheck(multiply_t multiply){
    printf("--------- Sanity Check ---------\n");
    // Sanity Check to see if multiplication works
    double** testMatrix = getIdentity(5);
    double* testVector = getVector(5);
    printf("Sanity Check: \n");
    printf("Matrix: \n");
    printMatrix(testMatrix, 5, 5);
    printf("Vector: \n");
    printVector(testVector, 5);
    printf("Product: \n");
    double* testResult = multiply(testMatrix, testVector, 5);
    printVector(testResult, 5);
    printf("--------------------------------\n");
}


/**
 * Runs the actual benchmark. Reads dimension and number of iterations from ENV vars.
 * @param multiply Function pointer to function with interface as defined in 'multiply_t'
*/
int runBenchmark(multiply_t multiply){
    // Load constants from env variables
    const int N = atoi(getenv("BENCHMARK_N"));
    const int nIterations = atoi(getenv("BENCHMARK_N_ITERATIONS"));
    if (!N){
        printf("Couldnt find 'BENCHMARK_N' in env variables.");
        return 1;
    } else if (!nIterations){
        printf("Couldnt find 'BENCHMARK_N_ITERATIONS' in env variables.");
        return 1;
    }

    // Construct Data
    double** identity = getIdentity(N);
    double* vector = getVector(N);

    // ---- Benchmark multiple iterations ----
    // Measure start time
    double start;
    if (approach == baseline){
        start = (double) clock();
    } else if (approach == omp) {
        start = omp_get_wtime();
    } else {
        printf("Approach %d not implemented", approach);
        return 1;
    }

    // Run benchmark
    for (int i = 0; i < nIterations; i++) {
        double* newVector = multiply(identity, vector, N);
        free(vector);   // Avoid memory leaks, since a new result vector is allocated in multiply
        vector = newVector;
    }

    // Measure end time
    double time;
    char* approachName;
    int nThreads; 

    if (approach == baseline) {
        time = (clock() - start) / (double) CLOCKS_PER_SEC;
        approachName = "baseline";
        nThreads = 1;
    } else if (approach == omp) {
        time = omp_get_wtime() - start;
        approachName = "omp";
        nThreads = omp_get_max_threads();
    } else {
        printf("Approach %d not implemented", approach);
        return 1;
    }

    // Print results
    printf("\n\n--------- Benchmark Results ---------\n");
    printf("Approach: %s\nNumber of Iterations: %d\nDimensionality: %d\nNumber of Threads: %d\nExecution Time: %.2fs\n",
            approachName, nIterations, N, nThreads, time);
    printf("-------------------------------------");
    return 0;
}