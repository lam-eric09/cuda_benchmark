/*
 * File: vector_add_double_pinned.cu
 * Description:
 *     This program demonstrates element-wise addition of two large double arrays 
 *     using CUDA, with pinned host memory and re-usage of allocated memory.
 *     The aim is to show the performance gain that can be achieved when addressing the
 *     major bottlenecks of a naive vector addition implementation in CUDA, which are:
 *     - Copy from host to device and vice versa when the memory allocated on host is pageable
 *     - Frequent allocation and deallocation of memory 
 * 
 * Usage:
 *     nvcc -arch=sm_50 vector_add_double_pinned.cu -o vector_add_double_pinned
 *     ./vector_add_double_pinned
 * 
 * Note:
 *     K defines the number of time the kernel function is called. Increase it to see how beneficial re-usage of memory is.
 *     N is the size of the double arrays.
 *     Be cautious of memory usage when setting large values for N.
 *     one double takes 8 bytes, so N = 1e08 takes 800MB.
 *     Three arrays are manipulated, taking a total of 2.4GB.
 */

#include <cstdlib>
#include <iostream>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cmath>

constexpr int K = 5;
constexpr int N = 100000000;

__global__ void add_vector(double *out, double *a, double *b){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<N) out[i] = a[i] + b[i];
}

void run_kernel(double *out, double *a, double *b, double *d_out, double *d_a, double *d_b, size_t bytes){
    // Initialize data
    for (int i=0; i<N; i++){
        a[i] = 1.0;
        b[i] = 2.0;
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

    // Calling the kernel K times
    for(int i=0; i<K; i++){
        run_kernel(out, a, b, d_out, d_a, d_b, bytes);
    }

    // Clean up 
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(out);
    return 0;
}
