#include <cstdlib>
#include <iostream>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cmath>

constexpr size_t N = 1000000;

__global__ void sub_vector(double *out, double *a, double *b){
    int i = threadIdx.x + blockIdx.x + blockDim.x;
    if (i<N) out[i] = a[i] + b[i];
}

int main(){
    double *out, *a, *b;
    int bytes = sizeof(double) * N;

    cudaMallocHost((void**)&out, bytes);
    cudaMallocHost((void**)&a, bytes);
    cudaMallocHost((void**)&b, bytes);

    for (int i=0; i<N; i++){
        a[i] = 1L;
        b[i] = 2L;
    }

    double *d_out, *d_a, *d_b;
    cudaMalloc((void**)&d_out, bytes);
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);
   
    int thread_per_block = 64;
    int total_blocks = ceil((N+thread_per_block-1)/thread_per_block);
    
    sub_vector<<<total_blocks,thread_per_block>>>(d_out, d_a, d_b);

    cudaFree(d_out);
    cudaFree(d_a);
    cudaFree(d_b);
    free(out);
    free(a);
    free(b);

    return 0;
}