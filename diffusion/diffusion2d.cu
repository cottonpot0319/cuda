#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ido.h"


#define  blockDim_x     256
#define  blockDim_y       8



__global__  void  cuda_diffusion2d_0
// ====================================================================
//
// program    :  CUDA device code for 2-D diffusion equation
//               for 16 x 16 block and 16 x 16 thread per 1 block
//
// date       :  Nov 07, 2008
// programmer :  Takayuki Aoki
// place      :  Tokyo Institute of Technology
//
(
   float    *f,         /* dependent variable                        */
   float    *fn,        /* dependent variable                        */
   int      nx,         /* grid number in the x-direction            */
   int      ny,         /* grid number in the x-direction            */
   float    c0,         /* coefficient no.0                          */
   float    c1,         /* coefficient no.1                          */
   float    c2          /* coefficient no.2                          */
)
// --------------------------------------------------------------------
{
   int    j,    jx,   jy;
   float  fcc,  fce,  fcw,  fcs,  fcn;

   jy = blockDim.y*blockIdx.y + threadIdx.y;
   jx = blockDim.x*blockIdx.x + threadIdx.x;
   j = nx*jy + jx;

   fcc = f[j];

   if(jx == 0) fcw = fcc;
   else        fcw = f[j - 1];

   if(jx == nx - 1) fce = fcc;
   else             fce = f[j+1];

   if(jy == 0) fcs = fcc;
   else        fcs = f[j-nx];

   if(jy == ny - 1) fcn = fcc;
   else             fcn = f[j+nx];

   fn[j] = c0*(fce + fcw)
         + c1*(fcn + fcs)
         + c2*fcc;
}



__global__  void  cuda_diffusion2d_3
// ====================================================================
//
// program    :  CUDA device code for 2-D diffusion equation
//               for 256 x blockDim_y block and blockDim_x thread
//               per 1 block
//
// date       :  May 13, 2009
// programmer :  Takayuki Aoki
// place      :  Tokyo Institute of Technology
//
(
   float    *f,         /* dependent variable                        */
   float    *fn,        /* dependent variable                        */
   int      nx,         /* grid number in the x-direction            */
   int      ny,         /* grid number in the x-direction            */
   float    c0,         /* coefficient no.0                          */
   float    c1,         /* coefficient no.1                          */
   float    c2          /* coefficient no.2                          */
)
// --------------------------------------------------------------------
{
     int    j,   jy;
     float  f0,  f1,  f2,  tmp;
     __shared__   float   fs[blockDim_x + 2];

     jy = blockDim_y*blockIdx.y;
     j = nx*jy + blockDim.x*blockIdx.x + threadIdx.x;
     f1 = f[j];

     if(blockIdx.y == 0) f0 = f1;
     else                f0 = f[j-nx];
     j += nx;

#pragma unroll

     for(jy = 0; jy < blockDim_y; jy++) {
         if(blockIdx.y == gridDim.y - 1) f2 = f1;
         else                            f2 = f[j];

         fs[threadIdx.x + 1] = f1;

         if(threadIdx.x == 0) {
            if(blockIdx.x == 0) fs[0] = f1;
            else                fs[0] = f[j-nx-1];
         }
         if(threadIdx.x == blockDim.x - 1) {
            if(blockIdx.x == gridDim.x - 1) fs[threadIdx.x + 2] = f1;
            else                            fs[threadIdx.x + 2] = f[j-nx+1];
         }

         __syncthreads();

         fn[j-nx] = c0*(fs[threadIdx.x] + fs[threadIdx.x+2])
               + c1*(f0 + f2)
               + c2*f1;

         j += nx;

         tmp = f0;  f0 = f1;  f1 = f2;  f2 = tmp;
     }
}


__global__  void  cuda_diffusion2d_2
// ====================================================================
//
// program    :  CUDA device code for 2-D diffusion equation
//               for 256 x blockDim_y block and blockDim_x thread
//               per 1 block
//
// date       :  May 13, 2009
// programmer :  Takayuki Aoki
// place      :  Tokyo Institute of Technology
//
(
   float    *f,         /* dependent variable                        */
   float    *fn,        /* dependent variable                        */
   int      nx,         /* grid number in the x-direction            */
   int      ny,         /* grid number in the x-direction            */
   float    c0,         /* coefficient no.0                          */
   float    c1,         /* coefficient no.1                          */
   float    c2          /* coefficient no.2                          */
)
// --------------------------------------------------------------------
{
     int    j,   js = threadIdx.x + 1,  jy,   j0 = 0,   j1 = 1,   j2 = 2,  tmp;
     __shared__   float   fs[3][blockDim_x + 2];

     jy = blockDim_y*blockIdx.y;
     j = nx*jy + blockDim.x*blockIdx.x + threadIdx.x;
     fs[j1][js] = f[j];

     if(blockIdx.y == 0) fs[j0][js] = fs[j1][js];
     else                fs[j0][js] = f[j-nx];
     j += nx;

#pragma unroll

     for(jy = 0; jy < blockDim_y; jy++) {
         if(blockIdx.y == gridDim.y - 1) fs[j2][js] = fs[j1][js];
         else                            fs[j2][js] = f[j];

         if(threadIdx.x == 0) {
            if(blockIdx.x == 0) fs[j1][0] = fs[j1][1];
            else                fs[j1][0] = f[j-nx-1];
         }
         if(threadIdx.x == blockDim.x - 1) {
            if(blockIdx.x == gridDim.x - 1) fs[j1][js + 1] = fs[j1][js];
            else                            fs[j1][js + 1] = f[j-nx+1];
         }

         __syncthreads();

         fn[j-nx] = c0*(fs[j1][js - 1] + fs[j1][js + 1])
               + c1*(fs[j0][js] + fs[j2][js])
               + c2*fs[j1][js];

         j += nx;

         tmp = j0;  j0 = j1;  j1 = j2;  j2 = tmp;
     }
}



void  cpu_diffusion2d
// ====================================================================
//
// purpos     :  2-dimensional diffusion equation solved by FDM
//
// date       :  May 16, 2008
// programmer :  Takayuki Aoki
// place      :  Tokyo Institute of Technology
//
(
   float    *f,         /* dependent variable                        */
   float    *fn,        /* updated dependent variable                */
   int      nx,         /* x-dimensional grid size                   */
   int      ny,         /* y-dimensional grid size                   */
   float    c0,         /* coefficient no.0                          */
   float    c1,         /* coefficient no.1                          */
   float    c2          /* coefficient no.2                          */
)
// --------------------------------------------------------------------
{
     int    j,    jx,   jy;
     float  fcc,  fce,  fcw,  fcs,  fcn;

     for(jy = 0; jy < ny; jy++) {
         for(jx = 0; jx < nx; jx++) {
             j = nx*jy + jx;
             fcc = f[j];

             if(jx == 0) fcw = fcc;
             else        fcw = f[j - 1];

             if(jx == nx - 1) fce = fcc;
             else             fce = f[j+1];

             if(jy == 0) fcs = fcc;
             else        fcs = f[j-nx];

             if(jy == ny - 1) fcn = fcc;
             else             fcn = f[j+nx];

             fn[j] = c0*(fce + fcw)
                   + c1*(fcn + fcs)
                   + c2*fcc;
         }
     }
}



float  diffusion2d
// ====================================================================
//
// purpos     :  2-dimensional diffusion equation solved by FDM
//
// date       :  May 16, 2008
// programmer :  Takayuki Aoki
// place      :  Tokyo Institute of Technology
//
(
   int      numGPUs,    /* available number of GPU device            */
   int      nx,         /* x-dimensional grid size                   */
   int      ny,         /* y-dimensional grid size                   */
   float    *f,         /* dependent variable                        */
   float    *fn,        /* updated dependent variable                */
   float    kappa,      /* diffusion coefficient                     */
   float    dt,         /* time step interval                        */
   float    dx,         /* grid spacing in the x-direction           */
   float    dy          /* grid spacing in the y-direction           */
)
// --------------------------------------------------------------------
{
     float     c0 = kappa*dt/(dx*dx),   c1 = kappa*dt/(dy*dy),
               c2 = 1.0 - 2.0*(c0 + c1);

     if(numGPUs > 0) {

        dim3  grid(nx/blockDim_x,ny/blockDim_y,1),  threads(blockDim_x,1,1);
        cuda_diffusion2d_3<<< grid, threads >>>(f,fn,nx,ny,c0,c1,c2);

        cudaThreadSynchronize();
     }
     else cpu_diffusion2d(f,fn,nx,ny,c0,c1,c2);

     return (float)(nx*ny)*7.0;
}
