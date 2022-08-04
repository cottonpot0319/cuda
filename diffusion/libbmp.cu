#ifndef SLS_put_header_bmp
#define SLS_put_header_bmp

#include <stdio.h>
#include <stdlib.h>

#include "ifr_t.h"

#ifndef SLS_fputc2lh
#define SLS_fputc2lh

#include <limits.h>

int fputc2lh(
unsigned short int d,
FILE           *sput)
{
  putc(d  &   0xFF,sput);
  return
  putc(d>>CHAR_BIT,sput);
}

#endif


#ifndef SLS_fputc4lh
#define SLS_fputc4lh


int fputc4lh(
unsigned long int d,
FILE          *sput)
{
  putc( d             &0xFF,sput);
  putc((d>>CHAR_BIT  )&0xFF,sput);
  putc((d>>CHAR_BIT*2)&0xFF,sput);
  return
  putc((d>>CHAR_BIT*3)&0xFF,sput);
}

#endif
// #include "fputc2lh.h"
// #include "fputc4lh.h"


int put_header_bmp(
/*
 * IFR_SUCCESS :
 * IFR_PUT_ERR :
 * IFR_SIZE_ERR:
 */
FILE *sput,
int     rx,
int     ry,
int   cbit)
{
  int i,   nx;
  int color;
  unsigned long int bfOffBits;

  if(rx<=0 || ry<=0)return IFR_SIZE_ERR;
  if(sput==NULL||ferror(sput))return IFR_PUT_ERR;
  nx = rx - (rx % 4);  if(nx < rx) nx += 4;

  if(cbit==24)color=0;else
  {
    color=1;
    for(i=1;i<=cbit;i++)color*=2;
  }
  bfOffBits= 14 +40 +4*color;

  fputs("BM",sput);
  fputc4lh(
    bfOffBits+(unsigned long)nx*ry,sput);
  fputc2lh(0,sput);
  fputc2lh(0,sput);
  fputc4lh(bfOffBits,sput);
  if(ferror(sput))return IFR_PUT_ERR;

  fputc4lh(  40,sput);
  fputc4lh(  rx,sput);
  fputc4lh(  ry,sput);
  fputc2lh(   1,sput);
  fputc2lh(cbit,sput);
  fputc4lh(   0,sput);
  fputc4lh(   0,sput);
  fputc4lh(   0,sput);
  fputc4lh(   0,sput);
  fputc4lh(   0,sput);
  fputc4lh(   0,sput);
  if(ferror(sput))return IFR_PUT_ERR;

  return IFR_SUCCESS;
}

#endif /* SLS_put_header_bmp */


int put_header_bmp(FILE *sput, int nx, int ny, int cbit);


void   DFR8bmp
// ====================================================================
//
// purpos     :  generation of 8-bit BMP format file (2-Dimensional)
//
// date       :  Jul 29, 2004
// programmer :  Takayuki Aoki
// place      :  Tokyo Institute of Technology
//
(
   char     *a,            /* data array to be output                */
   int      nx,            /* array size in the x-direction          */
   int      ny,            /* array size in the y-direction          */
   char     *filename,     /* raster 8 HDF data filename             */
   char     *palette       /* 256x3 pallete filename                 */
)
// --------------------------------------------------------------------
{
     int   j,  jx,  jy,  mx;
     unsigned  char   pal[768],  rgba[4] = {0,0,0,0},  *b;
     FILE  *fp = fopen(palette,"rb"), *fc = fopen(filename,"wb");
     if(fp != NULL) fread(pal,1,768,fp);

     put_header_bmp(fc,nx,ny,8);
     for(j = 0; j < 256; j++) {
         rgba[0] = pal[3*j+2];
         rgba[1] = pal[3*j+1];
         rgba[2] = pal[3*j+0];

         fwrite(rgba,1,4,fc);
     }

     mx = nx - (nx % 4);  if(mx < nx) mx += 4;
     b = (unsigned char *) calloc(mx*ny,sizeof(unsigned char));
     for(jy = 0; jy < ny; jy++) {
         for(jx = 0; jx < nx; jx++) b[jy*mx+jx] = a[(ny-1 - jy)*nx+jx];
     }
     fwrite(b,1,mx*ny,fc);

     if(fp != NULL) fclose(fp);
     fclose(fc);
     free(b);
}


void   dfr8bmp_
// ====================================================================
//
// purpos     :  FORTRAN rapper for DFR8bmp (in C)
//
// date       :  Aug 03, 2004
// programmer :  Takayuki Aoki
// place      :  Tokyo Institute of Technology
//
(
   char     *a,            /* data array to be output                */
   int      *nx,           /* array size in the x-direction          */
   int      *ny,           /* array size in the y-direction          */
   char     *filename,     /* raster 8 HDF data filename             */
   char     *palette       /* 256x3 pallete filename                 */
)
// --------------------------------------------------------------------
{
   DFR8bmp(a,*nx,*ny,filename,palette);
}


void   DFR8BMP_
// ====================================================================
//
// purpos     :  FORTRAN rapper for DFR8bmp (in C)
//
// date       :  Aug 03, 2004
// programmer :  Takayuki Aoki
// place      :  Tokyo Institute of Technology
//
(
   char     *a,            /* data array to be output                */
   int      *nx,           /* array size in the x-direction          */
   int      *ny,           /* array size in the y-direction          */
   char     *filename,     /* raster 8 HDF data filename             */
   char     *palette       /* 256x3 pallete filename                 */
)
// --------------------------------------------------------------------
{
   DFR8bmp(a,*nx,*ny,filename,palette);
}
