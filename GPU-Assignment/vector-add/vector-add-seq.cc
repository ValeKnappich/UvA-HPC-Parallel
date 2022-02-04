#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <chrono>

using namespace std::chrono;

void vectorAddSeq(int n, float* a, float* b, float* result) {
    int i;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    for (i=0; i<n; i++) {
        result[i] = a[i]+b[i];
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    std::cout << "vector-add (sequential): \t\t" << duration_cast<microseconds>(t2 - t1).count() << "us" << std::endl;
}

int main(int argc, char* argv[]) {
    int n = 655360;

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

    delete[] a;
    delete[] b;
    delete[] result;

    return 0;
}
