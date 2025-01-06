#include <cstdlib>
#include <iostream>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cmath>

#define N 100000000

__global__ void add_vector(double *out, double *a, double *b){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<N) out[i] = a[i] + b[i];
}

int main(){
    double *out, *a, *b;
    int bytes = N*sizeof(double);

    // Allocate pinned non pageable memory in host
    cudaMallocHost((void**)&out, bytes);
    cudaMallocHost((void**)&a, bytes);
    cudaMallocHost((void**)&b, bytes);

    // Allocate device memory
    double *d_out, *d_a, *d_b;
    cudaMalloc((void**)&d_out, bytes);
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);
    
    // Initialize data
    for (int i=0; i<N; i++){
        a[i] = 1L;
        b[i] = 2L;
    }
        
    // Copy to device
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

    // Run kernel 
    int thread_per_block = 64;
    int total_blocks = ceil((N+thread_per_block-1)/thread_per_block);
    add_vector<<<total_blocks,thread_per_block>>>(d_out, d_a, d_b);

    // Copy to host
    cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost);

    std::cout << N-1 << out[N-1] << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(out);
    return 0;
}
