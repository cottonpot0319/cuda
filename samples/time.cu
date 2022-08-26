#include <iostream>
#include <vector>
#include <time.h>
#include <cuda_runtime.h>

__global__
void matmul_naive(float *a, float *b, float *c, int matrix_size){
    const unsigned int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int yidx = blockIdx.y * blockDim.y + threadIdx.y;

    float accumulator = 0.0;
    for(int i = 0; i < matrix_size; i++)
        accumulator += a[yidx * matrix_size + i] * b[i * matrix_size + xidx];

    c[yidx * matrix_size + xidx] = accumulator;
}

__global__
void matmul_shared(float * a, float * b, float *c, int matrix_size){
    const unsigned int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int yidx = blockIdx.y * blockDim.y + threadIdx.y;

    float accumulator = 0.0;
    for(int i = 0; i < matrix_size; i+=16){
        __shared__ float sub_a[16][16];
        __shared__ float sub_b[16][16];

        sub_a[threadIdx.y][threadIdx.x] = a[yidx * matrix_size + (i + threadIdx.x)];
        sub_b[threadIdx.y][threadIdx.x] = b[(i + threadIdx.y) * matrix_size + xidx];

        __syncthreads();

        for(int j = 0; j < 16; j++)
            accumulator += sub_a[threadIdx.y][j] * sub_b[j][threadIdx.x];

        __syncthreads();
    }
    c[yidx * matrix_size + xidx] = accumulator;
}

int main(int argc, char **argv){
    //struct timespec tsStart, tsEnd;

    enum Mode{
        kModeNaive,
        kModeShared
    };

    enum Mode mode;

    if(2 != argc){
        std::cerr << "usage : ./time [naive | shared]" << std::endl;
        exit(-1);
    }

    if(0 == strcmp(argv[1], "naive")) mode = kModeNaive;
    else if(0 == strcmp(argv[1], "shared")) mode = kModeShared;
    else{
        std::cerr << "usage : ./time [naive | shared]" << std::endl;
        exit(-1);
    }

    const int matrix_size = 512; //must be multiple of 16
    const int size = matrix_size * matrix_size;
    std::vector<float> a(size);
    std::vector<float> b(size);
    std::vector<float> c(size);

    for(int i = 0; i < matrix_size; i++){
        for(int j = 0; j < matrix_size; j++){
            a[i * matrix_size + j] = i / static_cast<float>(matrix_size);
            b[i * matrix_size + j] = j / static_cast<float>(matrix_size);
        }
    }

    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, size * sizeof(float));
    cudaMalloc(&d_b, size * sizeof(float));
    cudaMalloc(&d_c, size * sizeof(float));

    cudaMemcpy(d_a, &a[0], size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b[0], size*sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid_size = dim3(matrix_size/16, matrix_size/16, 1);
    dim3 block_size = dim3(16, 16, 1);

    //timespec_get(&tsStart, TIME_UTC);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    switch(mode){
        case kModeNaive:
            matmul_naive<<<grid_size, block_size>>>(d_a, d_b, d_c, matrix_size);
            break;
        case kModeShared:
            matmul_shared<<<grid_size, block_size>>>(d_a, d_b, d_c, matrix_size);
            break;
        default:
            break;
    }
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    /*
    timespec_get(&tsEnd, TIME_UTC);
    int nsec = tsEnd.tv_nsec - tsStart.tv_nsec;
    int secSpan = tsEnd.tv_sec - tsStart.tv_sec;
    if(0 < secSpan) nsec += secSpan * 1000000000;
    */

    std::cout << milliseconds << " ms" << std::endl;

    cudaMemcpy(&c[0], d_c, size*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    /*
    for(int i = 0; i < matrix_size; i++){
        for(int j = 0; j < matrix_size; j++){
            printf("%5.2f ", c[i * matrix_size + j]);
        }
    } printf("\n");
    */

    return 0;
}