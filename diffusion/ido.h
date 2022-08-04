#define   MAX2(x, y) ((x) > (y) ? (x) : (y))
#define   MIN2(x, y) ((x) < (y) ? (x) : (y))


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
);
// --------------------------------------------------------------------


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
);
// --------------------------------------------------------------------


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
);
// --------------------------------------------------------------------


void   malloc_variables
// ====================================================================
//
// purpos     :  update the variable fn --> f
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
);
// --------------------------------------------------------------------


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
);
// --------------------------------------------------------------------


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
);
// --------------------------------------------------------------------


int     set_numGPUs(int numGPUs, int argc, char **argv);
