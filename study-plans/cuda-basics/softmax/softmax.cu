#include <cuda_runtime.h>
#include <math.h>

// The annotation __device__ means that this is a GPU helper function that is called by GPU
// result is written to sdata[0]
__device__ void reduce_max(float* sdata) {
    int tid = threadIdx.x;

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }
}

__device__ void reduce_sum(float* sdata) {
    int tid = threadIdx.x;

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
}

__global__ void softmax_kernel(const float* input, float* output, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    // Each threads scans multiple elements to find max elem within thread
    float threadMax = -INFINITY;
    for (int i = tid; i < N; i += blockDim.x) {
        threadMax = fmaxf(threadMax, input[i]);
    }

    // store thread max and find global max elem
    sdata[tid] = threadMax;
    __syncthreads();
    reduce_max(sdata);

    //Each threads perform the exp function and accumulate sum within thread
    float threadSum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        output[i] = expf(input[i] - sdata[0]);
        threadSum += output[i];
    }

    sdata[tid] = threadSum;
    __syncthreads();
    reduce_sum(sdata);

    for (int i = tid; i < N; i += blockDim.x) {
        output[i] /= sdata[0];
    } 
    
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 512;
    int blocks = 1;
    int shared_bytes = threads * sizeof(float);
    softmax_kernel<<<blocks, threads, shared_bytes>>>(input, output, N);
    cudaDeviceSynchronize();
}