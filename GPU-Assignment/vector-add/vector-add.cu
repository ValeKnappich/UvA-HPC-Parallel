#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <chrono>

using namespace std::chrono;

/* Utility function, use to do error checking.

   Use this function like this:

   checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));

   And to check the result of a kernel invocation:

   checkCudaCall(cudaGetLastError());
*/
static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "cuda error: " << cudaGetErrorString(result) << std::endl;
        exit(1);
    }
}

__global__ void vectorAddKernel(float* A, float* B, float* Result) {
    // insert operation here
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    Result[i] = A[i] + B[i];
}

void vectorAddCuda(int n, float* a, float* b, float* result) {
    int threadBlockSize;

    if (n < 512)
        threadBlockSize = n;
    else
        threadBlockSize = 512;

    // allocate the vectors on the GPU
    float* deviceA = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceA, n * sizeof(float)));
    if (deviceA == NULL) {
        std::cout << "could not allocate memory!" << std::endl;
        return;
    }
    float* deviceB = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceB, n * sizeof(float)));
    if (deviceB == NULL) {
        checkCudaCall(cudaFree(deviceA));
        std::cout << "could not allocate memory!" << std::endl;
        return;
    }
    float* deviceResult = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceResult, n * sizeof(float)));
    if (deviceResult == NULL) {
        checkCudaCall(cudaFree(deviceA));
        checkCudaCall(cudaFree(deviceB));
        std::cout << "could not allocate memory!" << std::endl;
        return;
    }

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    // copy the original vectors to the GPU
    checkCudaCall(cudaMemcpy(deviceA, a, n*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(deviceB, b, n*sizeof(float), cudaMemcpyHostToDevice));

    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    // execute kernel
    vectorAddKernel<<<n/threadBlockSize, threadBlockSize>>>(deviceA, deviceB, deviceResult);
    cudaDeviceSynchronize();

    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    high_resolution_clock::time_point t3 = high_resolution_clock::now();

    // copy result back
    checkCudaCall(cudaMemcpy(result, deviceResult, n * sizeof(float), cudaMemcpyDeviceToHost));

    high_resolution_clock::time_point t4 = high_resolution_clock::now();

    checkCudaCall(cudaFree(deviceA));
    checkCudaCall(cudaFree(deviceB));
    checkCudaCall(cudaFree(deviceResult));

    std::cout << "vector-add (H2D):    \t\t" << duration_cast<microseconds>(t2 - t1).count() << "us" << std::endl;
    std::cout << "vector-add (kernel): \t\t" << duration_cast<microseconds>(t3 - t2).count() << "us" << std::endl;
    std::cout << "vector-add (D2H):    \t\t" << duration_cast<microseconds>(t4 - t3).count() << "us" << std::endl;
}

void vectorAddSeq(int n, float* a, float* b, float* result) {
    int i;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    for (i=0; i<n; i++) {
        result[i] = a[i]+b[i];
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    std::cout << "vector-add (seq):    \t\t" << duration_cast<microseconds>(t2 - t1).count() << "us" << std::endl;
}

int main(int argc, char* argv[]) {
    // int n = 655360;
    int n = atoi(getenv("VECTOR_ADD_N"));
    float* a = new float[n];
    float* b = new float[n];
    float* result = new float[n];
    float* result_s = new float[n];

    if (argc > 1) n = atoi(argv[1]);

    std::cout << "Adding two vectors of " << n << " integer elements." << std::endl;

    // initialize the vectors.
    for(int i=0; i<n; i++) {
        a[i] = i;
        b[i] = i;
    }

    vectorAddSeq(n, a, b, result_s);
    vectorAddCuda(n, a, b, result);

    // verify the resuls
    for(int i=0; i<n; i++) {
        if (result[i]!=result_s[i]) {
            std::cout << "error in results! Element " << i << " is " << result[i] << ", but should be " << result_s[i] << std::endl;
            exit(1);
        }
    }

    std::cout << "results OK!" << std::endl;

    delete[] a;
    delete[] b;
    delete[] result;

    return 0;
}
