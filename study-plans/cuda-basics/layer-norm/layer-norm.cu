#include <cuda_runtime.h>
#include <math.h>

__device__ void reduce_sum(float* data) {
    int tid = threadIdx.x;

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            data[tid] += data[tid + stride];
        }
        __syncthreads();
    }
}

__global__ void layer_norm_kernel(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_varsum = sdata + blockDim.x; // This is already scaled by sizeof(float)
    
    int i = blockIdx.x;
    int j = threadIdx.x;

    // collect all mu components
    float threadSum = 0.0f;
    for (int col = j; col < N; col += blockDim.x) {
        threadSum += (input[i * N + col] / N);
    }
    s_sum[j] = threadSum;
    __syncthreads();
    // reduce sum mu components
    reduce_sum(s_sum);

    // Collect sigma squared components
    float threadVarSum = 0.0f;
    for (int col = j; col < N; col += blockDim.x) {
        float diff = input[i * N + col] - s_sum[0];
        threadVarSum += (diff * diff / N);
    } 
    s_varsum[j] = threadVarSum;
    __syncthreads();
    // reduce sum sigma squared components
    reduce_sum(s_varsum);

    // put everything together
    for (int col = j; col < N; col += blockDim.x) {
        output[i * N + col] = ((input[i * N + col] - s_sum[0]) / sqrtf(s_varsum[0] + eps)) * gamma[col] + beta[col];
    }
}

extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    int threads = 256;
    dim3 blocks(M);
    int shared_bytes = 2 * threads * sizeof(float);
    layer_norm_kernel<<<blocks, threads, shared_bytes>>>(input, gamma, beta, output, M, N, eps);
    cudaDeviceSynchronize();
}
