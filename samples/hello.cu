#include <iostream>
#include <stdio.h>

__global__ void hello(){
    printf("Hello, World!\n");
}

int main(int argc, char **argv){
    //関数名<<<ブロック数, スレッド数>>>(引数)
    //1グリッド、2ブロック、4スレッド=8並列？
    hello<<< 2, 4>>>();
    cudaDeviceSynchronize(); //GPUの処理が終わるまで待ってあげる👉MPI_Barrierみたいなもん？
    return 0;
}
