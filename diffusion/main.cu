// ---------------------------------------------------------------------
//
//     program : 2D diffusion equation solved by finite difference
//
//               Takayuki Aoki
//
//               Global Scientific Information and Computing Center
//               Tokyo Institute of Technology
//
//               2009, May 13
//
// ---------------------------------------------------------------------

#define  NX  256
#define  NY  256

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cutil.h>
#include "ido.h"


int  main(int argc, char *argv[])
{
     int      nx = NX,   ny = NY,   icnt = 1,   nout = 500,  numGPUs;
     float    *f,  *fn,  dt,   time = 0.0,  Lx = 1.0,  Ly = 1.0,
              dx = Lx/(float)nx,  dy = Ly/(float)ny,   kappa = 0.1,
              flops = 0.0;
     char     filename[] = "f000.bmp";

     cudaGetDeviceCount(&numGPUs);
     numGPUs = set_numGPUs(numGPUs,argc,argv);

     malloc_variables(numGPUs,nx,ny,&f,&fn);
     initial(numGPUs,nx,ny,dx,dy,f);
//   bmp_r8(numGPUs,nx,ny,f,1,1.11,-1.0,filename,"Seismic.pal");

     dt = 0.20*MIN2(dx*dx,dy*dy)/kappa;

     unsigned int  timer;
     cutCreateTimer(&timer);
     cutResetTimer(timer);
     cutStartTimer(timer);

     do {  if(icnt % 100 == 0) fprintf(stderr,"time(%4d)=%7.5f\n",icnt,time + dt);

           flops += diffusion2d(numGPUs,nx,ny,f,fn,kappa,dt,dx,dy);

           swap(&f,&fn);

           time += dt;

           if(icnt % nout == -1) {
              printf("TIME = %9.3e\n",time);
              sprintf(filename,"f%03d.bmp",icnt/nout);
              bmp_r8(numGPUs,nx,ny,f,1,1.11,-1.0,filename,"Seismic.pal");
           }

     } while(icnt++ < 99999 && time + 0.5*dt < 0.63);
     cutStopTimer(timer);
     float  elapsed_time = cutGetTimerValue(timer)*1.0e-03;

     printf("Elapsed Time= %9.3e [sec]\n",elapsed_time);
     printf("Performance= %7.2f [MFlops]\n",flops/elapsed_time*1.0e-06);

     return 0;
}



void   initial
// ====================================================================
//
// purpos     :  initial profile for variable f
//
// date       :  May 13, 2008
// programmer :  Takayuki Aoki
// place      :  Tokyo Institute of Technology
//
(
   int      numGPUs,    /* available number of GPU device            */
   int      nx,         /* x-dimension size                          */
   int      ny,         /* y-dimension size                          */
   float    dx,         /* grid spacing in the x-direction           */
   float    dy,         /* grid spacing in the y-direction           */
   float    *f          /* dependent variable f                      */
)
// --------------------------------------------------------------------
{
     int     j,    jx,   jy;
     float   *F,   x,    y,   alpha = 30.0;

     if(numGPUs > 0) F = (float *) malloc(nx*ny*sizeof(float));
     else            F = f;

     for(jy=0 ; jy < ny; jy++) {
         for(jx=0 ; jx < nx; jx++) {
             j = nx*jy + jx;
             x = dx*((float)jx + 0.5) - 0.5;   y = dy*((float)jy + 0.5) - 0.5;

             F[j] = exp(-alpha*(x*x + y*y));
         }
     }

     if(numGPUs > 0) {
        CUDA_SAFE_CALL( cudaMemcpy(f,F,nx*ny*sizeof(float), cudaMemcpyHostToDevice) );
        free(F);
     }
}
