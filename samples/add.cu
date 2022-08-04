#include <iostream>
#include <math.h>

__global__
void add(int n, float *x, float *y){
    int index = threadIdx.x;
    int stride = blockDim.x;
    for(int i = index; i < n; i+=stride)
        y[i] = x[i] + y[i];
}

int main(int argc, char **argv){
    int N = 1<<20; //1Mbyte elements = 2^20
    std::cout << N << std::endl;

    float *x, *y;

    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    for(int i = 0; i < N; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    add<<<1, 256>>>(N, x, y);

    cudaDeviceSynchronize();

    cudaFree(x);
    cudaFree(y);

    return 0;
}