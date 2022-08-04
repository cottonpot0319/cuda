__global__
void diff2dKernel(float* g_idata, float* g_idata2,int X, int Y)
{
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int bdim = blockDim.x;
    const unsigned int gdim = gridDim.x;
    int step=bdim*gdim;
    int num=X*Y;

    //ここで計算を行う
    const float Dfu=1;
    const float dt=0.2;
    const float dx=1;
    float Dfudtdx2=Dfu*dt/(dx*dx);
    for (int id=bid * bdim + tid;id<num;id+=step)
    {
        //境界は、nonflux境界条件
        if ((id)==0)//四隅
            g_idata2[id]=g_idata[id]+(g_idata[id+1]+g_idata[id+1]+g_idata[id+Y]+g_idata[id+Y]-4*g_idata[id])*Dfudtdx2;
        else if ((id)==Y-1)//四隅
            g_idata2[id]=g_idata[id]+(g_idata[id-1]+g_idata[id-1]+g_idata[id+Y]+g_idata[id+Y]-4*g_idata[id])*Dfudtdx2;
        else if ((id)==X*Y-Y)//四隅
            g_idata2[id]=g_idata[id]+(g_idata[id+1]+g_idata[id+1]+g_idata[id-Y]+g_idata[id-Y]-4*g_idata[id])*Dfudtdx2;
        else if ((id)==X*Y-1)//四隅
            g_idata2[id]=g_idata[id]+(g_idata[id-1]+g_idata[id-1]+g_idata[id-Y]+g_idata[id-Y]-4*g_idata[id])*Dfudtdx2;
        else if ((id)<Y)//四辺
            g_idata2[id]=g_idata[id]+(g_idata[id-1]+g_idata[id+1]+g_idata[id+Y]+g_idata[id+Y]-4*g_idata[id])*Dfudtdx2;
        else if((id)>X*Y-Y)//四辺
            g_idata2[id]=g_idata[id]+(g_idata[id-1]+g_idata[id+1]+g_idata[id-Y]+g_idata[id-Y]-4*g_idata[id])*Dfudtdx2;
        else if ((id)%Y==0)//四辺
            g_idata2[id]=g_idata[id]+(g_idata[id+1]+g_idata[id+1]+g_idata[id-Y]+g_idata[id+Y]-4*g_idata[id])*Dfudtdx2;
        else if ((id)%Y==Y-1)//四辺
            g_idata2[id]=g_idata[id]+(g_idata[id-1]+g_idata[id-1]+g_idata[id-Y]+g_idata[id+Y]-4*g_idata[id])*Dfudtdx2;
        else
            g_idata2[id]=g_idata[id]+(g_idata[id-1]+g_idata[id+1]+g_idata[id-Y]+g_idata[id+Y]-4*g_idata[id])*Dfudtdx2;
    }
}