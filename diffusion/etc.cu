#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cutil.h>
#include "ido.h"


void   swap
// ====================================================================
//
// purpos     :  update the variable fn --> f
//
// date       :  Jul 03, 2001
// programmer :  Takayuki Aoki
// place      :  Tokyo Institute of Technology
//
(
   float   **f,        /* dependent variable                        */
   float   **fn        /* updated variable                          */
)
// --------------------------------------------------------------------
{
     float  *tmp = *f;   *f = *fn;   *fn = tmp;
}



void   malloc_variables
// ====================================================================
//
// purpos     :  dynamic memory allocation for f and fn
//
// date       :  May 13, 2008
// programmer :  Takayuki Aoki
// place      :  Tokyo Institute of Technology
//
(
   int      numGPUs,    /* available number of GPU device            */
   int      nx,         /* x-size of the computational domain        */
   int      ny,         /* y-size of the computational domain        */
   float    **f,        /* variable f                                */
   float    **fn        /* updated variable fn                       */
)
// --------------------------------------------------------------------
{
     int   n = nx*ny;

     if(numGPUs > 0) {
        CUDA_SAFE_CALL( cudaMalloc( (void**) f,  nx*ny*sizeof(float)));
        CUDA_SAFE_CALL( cudaMalloc( (void**) fn, nx*ny*sizeof(float)));
     }
     else {
        *f  = (float *) malloc(sizeof(float)*n);
        *fn = (float *) malloc(sizeof(float)*n);
     }
}



void   bmp_r8
// ====================================================================
//
// purpos     :  generation of BMP image file (2-Dimensional)
//
// date       :  Aug 09, 2004
// programmer :  Takayuki Aoki
// place      :  Tokyo Institute of Technology
//
(
   int      numGPUs,       /* available number of GPU device         */
   int      nx,            /* x-dimension size                       */
   int      ny,            /* y-dimension size                       */
   float    *f,            /* dependent variable                     */
   int      mul,           /* multiple scale                         */
   float    fmax,          /* maximum value for f                    */
   float    fmin,          /* minimum value for f                    */
   char     *filename,     /* raster 8 BMP data filename             */
   char     *palette       /* 256x3 pallete filename                 */
)
// --------------------------------------------------------------------
{
     int     j,   jx,   jy,   ix,  iy,   jmx,  jmy,  mul_x,  mul_y;
     float  gux,  guy,  fc,  gmax=-1.0e+30,  gmin=1.0e+30,   f0,   f1,  *F;
     static  int  mx = 0,     my = 0;
     char   *a;

     mx = mul*(nx - 1) + 1;   my = mul*(ny - 1) + 1;
     a = (char *) malloc( sizeof(float)*mx*my);

     if(numGPUs > 0) {
       F = (float *) malloc(nx*ny*sizeof(float));
       cudaMemcpy( F, f, nx*ny*sizeof(float), cudaMemcpyDeviceToHost );
     }
     else F = f;


     for(jy=0; jy < ny-1; jy++) {
         for(jx=0; jx < nx-1; jx++) {
             j = nx*jy + jx;

             if(jx == nx - 1) mul_x = 1;  else mul_x = mul;
             if(jy == ny - 1) mul_y = 1;  else mul_y = mul;
             for(jmy=0; jmy < mul_y; jmy++) {
                 iy = mul*jy + jmy;
                 iy = my - 1 - iy;
                 for(jmx=0; jmx < mul_x; jmx++) {
                     ix = mul*jx + jmx;
                     gux = (float)jmx/(float)mul;
                     guy = (float)jmy/(float)mul;

                     f0 = (1.0 - gux)*F[j] + gux*F[j+1];
                     f1 = (1.0 - gux)*F[j+nx] + gux*F[j+nx+1];
                     fc = (1.0 - guy)*f0 + guy*f1;

                     gmax = MAX2(gmax, fc);
                     gmin = MIN2(gmin, fc);
                     fc = 253.0*(fc - fmin)/(fmax - fmin) + 2.0;

                     a[mx*iy + ix] = (char) MIN2( MAX2(2.0, fc), 253.0 );
                 }
             }
         }
     }

     DFR8bmp(a, mx, my, filename, palette);
     fprintf(stderr,"filename=%s  ",filename);
     fprintf(stderr,"MAX=%10.3e  MIN=%10.3e\n",gmax, gmin);

     free(a);
     if(numGPUs > 0) free(F);

     return;
}



int  set_numGPUs(int numGPUs, int argc, char **argv)
{
     int   usrGPUs;

     if(argc > 1) {
        if(strncmp(argv[1],"-gpu",4) == 0) usrGPUs = numGPUs;
        else if(strncmp(argv[1],"-cpu",4) == 0) usrGPUs = 0;
        else                               usrGPUs = numGPUs;
     }
     else  usrGPUs = numGPUs;

     return usrGPUs;
}
