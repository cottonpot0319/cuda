#include <stdio.h>
#include <cuda_runtime.h>

#include "diff2d_kernel.cu"

//系のサイズ 100 X 100のグリッド上で拡散方程式を解く
const int X=100;
const int Y=100;

int main( int argc, char** argv){
    //デバイスの初期化
    //CUT_DEVICE_INIT(argc, argv);

    //結果書き込み用ファイルのオープン
    FILE *fp=fopen("result.txt","w");

    //タイマーを作成して計測開始
    //unsigned int timer = 0;
    //cutCreateTimer(&timer);
    //cutStartTimer(timer);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    //メインメモリ上にfloat型のデータをX*Y個生成する
    float* h_idata = (float*) malloc(sizeof( float) * X*Y);
    //初期条件をセット
    for( int i = 0; i < X; i++) 
        for( int j = 0; j < X; j++)
            if((i-X/2)*(i-X/2)+(j-Y/2)*(j-Y/2)<10*10)
                h_idata[i*Y+j] = 1;
            else
                h_idata[i*Y+j] = 0;

    //デバイス上（ビデオカードのこと）にも同じくfloat型X*Y個分のメモリを確保する
    float* d_idata;
    cudaMalloc((void**) &d_idata, sizeof( float) * X*Y);
    //デバイス上（ビデオカードのこと）にfloat型X*Y個分の作業用メモリを確保する
    float* d_idata2;
    cudaMalloc((void**) &d_idata2, sizeof( float) * X*Y);

    //ブロック数を増やして並列度を上げる
    dim3  grid( 16, 1, 1);
    dim3  threads(256, 1, 1);
    
    //メインメモリからデバイスのメモリにデータを転送する
    cudaMemcpy( d_idata, h_idata, sizeof( float) * X*Y , cudaMemcpyHostToDevice);

    for (int t=0;t<100;t++){
        for (int n=0;n<10;n++){
            //ここでGPUを使った計算が行われる
            diff2dKernel<<< grid, threads>>>( d_idata, d_idata2,X,Y);
            //作業用領域から書き戻す
            cudaMemcpy( d_idata, d_idata2, sizeof( float) * X*Y, cudaMemcpyDeviceToDevice);
        }
        //デバイスからメインメモリ上に実行結果をコピー
        cudaMemcpy( h_idata, d_idata, sizeof( float) * X*Y, cudaMemcpyDeviceToHost);
        //実行結果を表示
        for (int i=0;i<X;i+=2){
            for (int j=0;j<Y;j+=2){
                fprintf(fp,"%f\t",h_idata[i*Y+j]);
            }
        }
        fprintf(fp,"\n");
    }
    //タイマーを停止しかかった時間を表示
    //cutStopTimer(timer);
    //printf("Processing time: %f (ms)\n", cutGetTimerValue(timer));
    //cutDeleteTimer(timer);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("time: %8.2f ms\n", milliseconds);

    //各種メモリを解放
    free(h_idata);
    cudaFree(d_idata);
    cudaFree(d_idata2);
    fclose(fp);
    //終了処理
    //CUT_EXIT(argc, argv);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
