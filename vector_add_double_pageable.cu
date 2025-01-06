#include <cstdlib>
#include <vector>
#include <iostream>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cmath>

constexpr int N = 100000000;

__global__ void vector_add(double *out, double *a, double *b){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<N) out[i] = a[i] + b[i];
}

int main(){
    int bytes = sizeof(double)*N;

    // Allocate pageable memory
    // a = new double[bytes];
    std::vector<double> a(N);
    std::vector<double> b(N);
    std::vector<double> out(N);

    for (int i=0; i<N; i++){
        a[i] = 1.0;
        b[i] = 2.0;
    }

    // Allocate device memory
    double *d_out, *d_a, *d_b;
    cudaMalloc((void**)&d_out, bytes);
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);

    // Copy to device
    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    // Run kernel
    int thread_per_block = 64;
    int total_blocks = ceil((N+thread_per_block-1)/thread_per_block);
    vector_add<<<total_blocks,thread_per_block>>>(d_out, d_a, d_b);

    cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    std::cout << N-1 << out[N-1] << std::endl;

    cudaFree(d_out);
    cudaFree(d_a);
    cudaFree(d_b);
    
    return 0;
}