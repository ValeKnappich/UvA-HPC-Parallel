#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include "timer.h"

using namespace std;


// const int key = 1; // = atoi(getenv("KEY"));


/* Utility function, use to do error checking.

   Use this function like this:

   checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));

   And to check the result of a kernel invocation:

   checkCudaCall(cudaGetLastError());
*/
static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        cerr << "cuda error: " << cudaGetErrorString(result) << endl;
        exit(1);
    }
}


__global__ void encryptKernel(int n, char* deviceDataIn, char* deviceDataOut, int key[], int keyLength) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned keyIndex = index % keyLength;
    if (index < n) 
        deviceDataOut[index] = deviceDataIn[index] + key[keyIndex];
}

__global__ void decryptKernel(int n, char* deviceDataIn, char* deviceDataOut, int key[], int keyLength) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned keyIndex = index % keyLength;
    if (index < n) deviceDataOut[index] = deviceDataIn[index] - key[keyIndex];
}

int fileSize(char* data_name) {
  int size; 

  // ifstream file ("original.data", ios::in|ios::binary|ios::ate);
  ifstream file (data_name, ios::in|ios::binary|ios::ate);
  if (file.is_open())
  {
    size = file.tellg();
    file.close();
  }
  else {
    cout << "Unable to open file";
    size = -1; 
  }
  return size; 
}

int readData(char *fileName, char *data) {

  streampos size;

  ifstream file (fileName, ios::in|ios::binary|ios::ate);
  if (file.is_open())
  {
    size = file.tellg();
    file.seekg (0, ios::beg);
    file.read (data, size);
    file.close();

    cout << "The entire file content is in memory." << endl;
  }
  else cout << "Unable to open file" << endl;
  return 0;
}

int writeData(int size, char *fileName, char *data) {
  ofstream file (fileName, ios::out|ios::binary|ios::trunc);
  if (file.is_open())
  {
    file.write (data, size);
    file.close();

    cout << "The entire file content was written to file." << endl;
    return 0;
  }
  else cout << "Unable to open file";

  return -1; 
}

int EncryptSeq (int n, char* data_in, char* data_out, int* key, int keyLength) 
{  
  timer sequentialTime = timer("Sequential encryption");
  sequentialTime.start();
  for (int i=0; i<n; i++) { 
    data_out[i]=data_in[i] + key[i % keyLength]; 
  }
  sequentialTime.stop();

  cout << fixed << setprecision(6);
  cout << "Encryption (sequential): \t\t" << sequentialTime.getElapsed() << " seconds." << endl;
  
  return 0; 
}

int DecryptSeq (int n, char* data_in, char* data_out, int* key, int keyLength)
{
  timer sequentialTime = timer("Sequential decryption");
  sequentialTime.start();
  for (int i=0; i<n; i++) { 
      data_out[i]=data_in[i] - key[i % keyLength]; 
  }
  sequentialTime.stop();

  cout << fixed << setprecision(6);
  cout << "Decryption (sequential): \t\t" << sequentialTime.getElapsed() << " seconds." << endl;

  return 0;
}

int EncryptCuda (int n, char* data_in, char* data_out, int* key, int keyLength) {
    int threadBlockSize = 512;
    // allocate the vectors on the GPU
    char* deviceDataIn = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceDataIn, n * sizeof(char)));
    if (deviceDataIn == NULL) {
        cout << "could not allocate memory!" << endl;
        return -1;
    }
    char* deviceDataOut = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceDataOut, n * sizeof(char)));
    if (deviceDataOut == NULL) {
        checkCudaCall(cudaFree(deviceDataIn));
        cout << "could not allocate memory!" << endl;
        return -1;
    }
    int* deviceKey = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceKey, keyLength * sizeof(int)));
    if (deviceKey == NULL) {
        checkCudaCall(cudaFree(deviceKey));
        cout << "could not allocate memory!" << endl;
        return -1;
    }

    timer kernelTime1 = timer("kernelTime");
    timer memoryTime = timer("memoryTime");

    // copy the original vectors to the GPU
    memoryTime.start();
    checkCudaCall(cudaMemcpy(deviceDataIn, data_in, n*sizeof(char), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(deviceKey, key, keyLength * sizeof(int), cudaMemcpyHostToDevice));
    memoryTime.stop();

    // execute kernel
    kernelTime1.start();
    encryptKernel<<<n/threadBlockSize+1, threadBlockSize>>>(n, deviceDataIn, deviceDataOut, deviceKey, keyLength);
    cudaDeviceSynchronize();
    kernelTime1.stop();

    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    // copy result back
    memoryTime.start();
    checkCudaCall(cudaMemcpy(data_out, deviceDataOut, n * sizeof(char), cudaMemcpyDeviceToHost));
    memoryTime.stop();

    checkCudaCall(cudaFree(deviceDataIn));
    checkCudaCall(cudaFree(deviceDataOut));
    checkCudaCall(cudaFree(deviceKey));

    cout << fixed << setprecision(6);
    cout << "Encrypt (kernel): \t\t" << kernelTime1.getElapsed() << " seconds." << endl;
    cout << "Encrypt (memory): \t\t" << memoryTime.getElapsed() << " seconds." << endl;

   return 0;
}

int DecryptCuda (int n, char* data_in, char* data_out, int* key, int keyLength) {
    int threadBlockSize = 512;

    // allocate the vectors on the GPU
    char* deviceDataIn = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceDataIn, n * sizeof(char)));
    if (deviceDataIn == NULL) {
        cout << "could not allocate memory!" << endl;
        return -1;
    }
    char* deviceDataOut = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceDataOut, n * sizeof(char)));
    if (deviceDataOut == NULL) {
        checkCudaCall(cudaFree(deviceDataIn));
        cout << "could not allocate memory!" << endl;
        return -1;
    }    
    int* deviceKey = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceKey, keyLength * sizeof(int)));
    if (deviceKey == NULL) {
        checkCudaCall(cudaFree(deviceKey));
        cout << "could not allocate memory!" << endl;
        return -1;
    }

    timer kernelTime1 = timer("kernelTime");
    timer memoryTime = timer("memoryTime");

    // copy the original vectors to the GPU
    memoryTime.start();
    checkCudaCall(cudaMemcpy(deviceDataIn, data_in, n*sizeof(char), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(deviceKey, key, keyLength * sizeof(int), cudaMemcpyHostToDevice));
    memoryTime.stop();

    // execute kernel
    kernelTime1.start();
    decryptKernel<<<n/threadBlockSize+1, threadBlockSize>>>(n, deviceDataIn, deviceDataOut, deviceKey, keyLength);
    cudaDeviceSynchronize();
    kernelTime1.stop();

    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    // copy result back
    memoryTime.start();
    checkCudaCall(cudaMemcpy(data_out, deviceDataOut, n * sizeof(char), cudaMemcpyDeviceToHost));
    memoryTime.stop();

    checkCudaCall(cudaFree(deviceDataIn));
    checkCudaCall(cudaFree(deviceDataOut));
    checkCudaCall(cudaFree(deviceKey));

    cout << fixed << setprecision(6);
    cout << "Decrypt (kernel): \t\t" << kernelTime1.getElapsed() << " seconds." << endl;
    cout << "Decrypt (memory): \t\t" << memoryTime.getElapsed() << " seconds." << endl;

   return 0;
}

int main(int argc, char* argv[]) {
    char* data_name = "text50000.data";

    int n = fileSize(data_name);
    if (n == -1) {
      cout << "File not found! Exiting ... " << endl; 
      exit(0);
    }

    char* data_in = new char[n];
    char* data_out = new char[n];    
    readData(data_name, data_in); 

    cout << "Encrypting a file of " << n << " characters." << endl;

    int key[] = {1, 9, 18, 42, 255, 424, 4, 242, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 4, 24, 2, 42, 4, 24, 24, 2, 4};
    int keyLength = sizeof(key) / sizeof(key[0]);

    EncryptSeq(n, data_in, data_out, key, keyLength);
    writeData(n, "sequential.data", data_out);
    EncryptCuda(n, data_in, data_out, key, keyLength);
    writeData(n, "cuda.data", data_out);  

    readData("cuda.data", data_in);

    cout << "Decrypting a file of " << n << "characters" << endl;
    DecryptSeq(n, data_in, data_out, key, keyLength);
    writeData(n, "sequential_decrypted.data", data_out);
    DecryptCuda(n, data_in, data_out, key, keyLength); 
    writeData(n, "recovered.data", data_out); 
 

    // load decrypted data and compare results
    char* data_seq = new char[n];
    char* data_cuda = new char[n];
    readData("sequential_decrypted.data", data_seq); 
    readData("recovered.data", data_cuda);     

    char* data_original = new char[n];
    readData("original.data", data_original); 

    // char difffe = diff(data_seq, data_cuda); // diff doesnt work

    cout << "Compare decrypted files" << endl;
    for (int i=0; i<n; i++) { 
      if(data_seq[i] != data_cuda[i]) {
          cout << "Files differ - i:" << i << " data_original[i]:" << data_original[i] << " data_seq[i]:" << data_seq[i] << " data_cuda[i]: " << data_cuda[i] << endl;
      }
    }

    // load decrypted data and compare results
    // char* data_seq = new char[n];
    // char* data_cuda = new char[n];
    // readData("sequential.data", data_seq); 
    // readData("cuda.data", data_cuda); 

    delete[] data_in;
    delete[] data_out;
    delete[] data_seq;
    delete[] data_cuda;
    delete[] data_original;


    return 0;
}
