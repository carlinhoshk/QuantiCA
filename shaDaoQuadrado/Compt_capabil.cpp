#include <stdio.h>
#include <assert.h>

#define NUMTHREADSPERBLOCK 10

__constant__ int kte_dev [NUMTHREADSPERBLOCK];
 int kte_host[NUMTHREADSPERBLOCK];

void __device__ addFunc(int idx, int *data) {
 data[idx] += kte_dev[idx];
}
__global__ void myKernel(int *deviceVar) {
 extern __shared__ int s_data[];
 // This is not necessary because we have only one block full of
 // threads but it's illustrative (blockDim.x = numThreadsPerBlock)
 // (blockDim.x = numThreadsPerBlock)*(blockIdx.x = 0) = 0
 int idx = blockDim.x *blockIdx.x + threadIdx.x;
 // Transfer data to shared memory
 s_data[idx] = deviceVar[idx];
 // Execute the addition function
 addFunc(idx, s_data);
 // Transfer data from shared memory
 deviceVar[idx] = s_data[idx];
}
int main(int argc, char **argv) {
 // Allocated in the host memory
 int *hostVar;
 int numThreadsPerBlock = NUMTHREADSPERBLOCK;
 int memSize = numThreadsPerBlock * sizeof(int);

 // Initialize and copy it to Constant Cache
 for(int i = 0; i < numThreadsPerBlock; i++) {
 kte_host[i] = i;
 }
 cudaMemcpyToSymbol(kte_dev, kte_host, sizeof(kte_host));
 // Allocated in the device memory
 int *deviceVar;
 // Alloc memory in host
 hostVar = (int *)malloc(memSize);
 cudaMalloc((void **)&deviceVar, memSize);
 // Fill with 0's
 memset(hostVar,0,memSize);
 // Copy it to device's memory
 cudaMemcpy(deviceVar, hostVar, memSize, cudaMemcpyHostToDevice);
 // Launch kernel 1 block NUMTHREADSPERBLOCK
 myKernel<<1,numThreadsPerBlock,memSize>>(deviceVar);
 // Wait until process it's finished
 // This is not necessary because the cudaMemcpy below is a blocking
 // function which will stop running the CPU task until deviceVar is
 // unblocked
 cudaThreadSynchronize();
 // Copy it back to host's memory
 cudaMemcpy(hostVar, deviceVar, memSize, cudaMemcpyDeviceToHost);
 // Test if something was wrong
 for(int i = 0; i < numThreadsPerBlock ; i++) {
 assert(hostVar[i] == i);
 }
 // Free host's memory
 free(hostVar);
}
