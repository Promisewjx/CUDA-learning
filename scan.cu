#include<common.h>
#include<timer.h>

#define BLOCK_DIM 1024

// kogge stone
__global__ void scan_kernel(float* input,float* output,float* partialSums,unsigned int N){

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float buffer_s[BLOCK_DIM];

    buffer_s[threadIdx.X] = input[I];

    for(const int stride = 1;stride <= BLOCK_DIM / 2;stride *= 2){
        if(threadIdx.x >= stride){
            float temp;
            temp =  buffer_s[threadIdx.x] + buffer_s[threadIdx - stride];
            __syncthreads();
            buffer_s[threadIdx.x] = temp;
        }
           
        __syncthreads();
    }

    if(threadIdx.x == BLOCK_DIM - 1)
        partialSums[blockIdx.x] = buffer_s[threadIdx.x];

    output[i] = buffer_s[threadIdx.x];
}

// 利用双缓冲解决读写冲突假依赖
__global__ void scan_kernel_db(float* input,float* output,float* partialSums,unsigned int N){

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float buffer1_s[BLOCK_DIM];
    __shared__ float buffer2_s[BLOCK_DIM];

    float* inBuffer_s = buffer1_s;
    float* outBuffer_s = buffer2_s;

    inBuffer_s[threadIdx.X] = input[I];
    __syncthreads();

    for(const int stride = 1;stride <= BLOCK_DIM / 2;stride *= 2){
        if(threadIdx.x >= stride){
            outBuffer_s[threadIdx.x] =  inBuffer_s[threadIdx.x] + inBuffer_s[threadIdx - stride];
        }else   
            outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x];
           
        __syncthreads();
        float* temp = inBuffer_s;
        inBuffer_s = outBuffer_s;
        outBuffer_s = temp;
    }

    if(threadIdx.x == BLOCK_DIM - 1)
        partialSums[blockIdx.x] = inBuffer_s[threadIdx.x];

    output[i] = inBuffer_s[threadIdx.x];
}

// brent kung reduce control divergence
__global__ void scan_kernel_bk(Float* input, Float* output, Float* partialSums, unsigned int N) {
    unsigned int segment = blockIdx.x * blockDim.x * 2;
    __shared__ float buffer_s[2 * BLOCK_DIM];

    buffer_s[threadIdx.x] = input[segment + threadIdx.x];
    buffer_s[threadIdx.x + BLOCK_DIM] = input[segment + threadIdx.x + BLOCK_DIM];
    __syncthreads();

    // Reduction phase
    for (unsigned int stride = 1; stride <= BLOCK_DIM; stride *= 2) {
        unsigned int i = (threadIdx.x + 1) * 2 * stride - 1;
        if (i < 2 * BLOCK_DIM) {
            buffer_s[i] += buffer_s[i - stride];
        }
        __syncthreads();
    }

    // Post reduction
    for (unsigned int stride = BLOCK_DIM / 2; stride >= 1; stride /= 2) {
        unsigned int i = (threadIdx.x + 1) * 2 * stride - 1;
        if (i + stride < 2 * BLOCK_DIM) {
            buffer_s[i + stride] += buffer_s[i];
        }
        __syncthreads();
    }

    if (threadIdx.x == BLOCK_DIM - 1) {
        partialSums[blockIdx.x] = buffer_s[2 * BLOCK_DIM - 1];
    }

    output[segment + threadIdx.x] = buffer_s[threadIdx.x];
    output[segment + threadIdx.x + BLOCK_DIM] = buffer_s[threadIdx.x + BLOCK_DIM];
}

// segmented scan
__global__ void scan_kernel(float* input, float* output, float* partialSums, unsigned int N) {
    unsigned int segment = blockIdx.x * blockDim.x * COARSE_FACTOR;

    // Load elements from global memory to shared memory
    __shared__ float buffer_s[BLOCK_DIM * COARSE_FACTOR];
    for(unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        buffer_s[c * BLOCK_DIM + threadIdx.x] = input[segment + c * BLOCK_DIM + threadIdx.x];
    }
    __syncthreads();

    // Thread scan
    unsigned int threadSegment = threadIdx.x * COARSE_FACTOR;
    for(unsigned int c = 1; c < COARSE_FACTOR; ++c) {
        buffer_s[threadSegment + c] += buffer_s[threadSegment + c - 1];
    }
    __syncthreads();

    // Allocate and initialize double buffers for partial sums
    __shared__ float buffer1_s[BLOCK_DIM];
    __shared__ float buffer2_s[BLOCK_DIM];
    float* inBuffer_s = buffer1_s;
    float* outBuffer_s = buffer2_s;

    inBuffer_s[threadIdx.x] = buffer_s[threadSegment + COARSE_FACTOR - 1];
    __syncthreads();

    // Parallel scan of partial sums
    for(unsigned int stride = 1; stride <= BLOCK_DIM / 2; stride *= 2) {
        if(threadIdx.x >= stride) {
            outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x] + inBuffer_s[threadIdx.x - stride];
        } else {
            outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x];
        }
        __syncthreads();
        float* tmp = inBuffer_s;
        inBuffer_s = outBuffer_s;
        outBuffer_s = tmp;
    }

    // Add previous thread's partial sum
    if (threadIdx.x > 0) {
        float prevPartialSum = inBuffer_s[threadIdx.x - 1];
        for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
            buffer_s[threadSegment + c] += prevPartialSum;
        }
    }
    __syncthreads();

    // Save block's partial sum
    if (threadIdx.x == BLOCK_DIM - 1) {
        partialSums[blockIdx.x] = inBuffer_s[threadIdx.x];
    }

    // Write output
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        output[segment + c * BLOCK_DIM + threadIdx.x] = buffer_s[c * BLOCK_DIM + threadIdx.x];
    }

}

// single kernel scan
