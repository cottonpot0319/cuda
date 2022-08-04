#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1600

struct Matrix{
    int width;
    int height;
    int stride;
    double* elements;
};

__device__ 
Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width  = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride*BLOCK_SIZE*row+BLOCK_SIZE*col];

    return Asub;
}

__global__ 
void matrixMulShared(Matrix A, Matrix B, Matrix C)
{
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if(row < C.height && col < C.width){
        int brow = blockIdx.y;
        int bcol = blockIdx.x;

        Matrix Csub = GetSubMatrix(C, brow, bcol);

        int trow = threadIdx.y;
        int tcol = threadIdx.x;

        float x = 0.0f;
        for(int l = 0; l < (A.width+BLOCK_SIZE-1)/BLOCK_SIZE; ++l){
            Matrix Asub = GetSubMatrix(A, brow, l);
            Matrix Bsub = GetSubMatrix(B, l, bcol);

            __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
            __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

            // サブ行列の内容をシェアードメモリへ
            As[trow][tcol] = Asub.elements[trow*Asub.stride+tcol];
            Bs[trow][tcol] = Bsub.elements[trow*Bsub.stride+tcol];

            // 他のスレッドがシェアードへの書き込みを終了するのを待つ
            __syncthreads();

            for(int k = 0; k < BLOCK_SIZE; ++k){
                x += As[trow][k]*Bs[k][tcol];
            }

            // 次の反復でサブ行列が書き換えられるため，ここで処理待ち
            __syncthreads();
        }

        Csub.elements[trow*Csub.stride+tcol] = x;
    }
}

int main(int argc, char** argv){
    //結果書き込み用ファイルのオープン
    //FILE *fp=fopen("result.txt","w");

    //ホスト側の行列の定義(サイズはブロックサイズの倍数に設定)
    Matrix hA, hB, hC;
    hA.height = hC.height = 3 * BLOCK_SIZE;
    hA.width  = hB.height = 4 * BLOCK_SIZE;
    hB.width  = hC.width  = 5 * BLOCK_SIZE;
    hA.elements = new double[hA.width * hA.height];
    hB.elements = new double[hB.width * hB.height];
    hC.elements = new double[hC.width * hC.height];
    for(int i = 0; i < hA.height*hA.width; i++) hA.elements[i] = 1.0;
    for(int i = 0; i < hB.height*hB.width; i++) hB.elements[i] = 2.0;

    //デバイス側のメモリ確保とデータ転送
    Matrix dA, dB, dC;
    dA.width = hA.width;    dA.height = hA.height;
    dB.width = hB.width;    dB.height = hB.height;
    dC.width = hC.width;    dC.height = hC.height;
    int size;
    // デバイスメモリの確保とホストからの転送
    size = dA.width*dA.height*sizeof(double);
    cudaMalloc((void**)&dA.elements, size);
    cudaMemcpy(dA.elements, hA.elements, size, cudaMemcpyHostToDevice);
    size = dB.width*dB.height*sizeof(double);
    cudaMalloc((void**)&dB.elements, size);
    cudaMemcpy(dB.elements, hB.elements, size, cudaMemcpyHostToDevice);
    size = dC.width*dC.height*sizeof(double);
    cudaMalloc((void**)&dC.elements, size);

    //カーネルの実行
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((dC.width+block.x-1)/block.x, (dC.height+block.y-1)/block.y);
    matrixMulShared<<<grid, block >>>(dA, dB, dC);
    // カーネル実行エラーのチェック
    //cutilCheckMsg("Kernel execution failed");

    // デバイスからホストへ結果を転送
    size = dC.width*dC.height*sizeof(double);
    cudaMemcpy(hC.elements, dC.elements, size, cudaMemcpyDeviceToHost);

    //タイマーを作成して計測開始
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    //タイマーを停止しかかった時間を表示
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("time: %8.3f ms\n", milliseconds);

    //計算結果
    //for(int i = 0; i < hC.height*hC.width; i++) std::cout << hC.elements[i] << std::endl; 

    // デバイスメモリ解放
    cudaFree(dA.elements);
    cudaFree(dB.elements);
    cudaFree(dC.elements);
    // ホストメモリ解放
    delete [] hA.elements;
    delete [] hB.elements;
    delete [] hC.elements;
    //fclose(fp);

    //終了処理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
