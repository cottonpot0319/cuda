# NFLAG = -D__DEVICE_EMULATION__  -deviceemu -g

CC = gcc
NVCC = nvcc
RM = /bin/rm
PROG = a.out

OBJS = diffusion2d.o etc.o main.o libbmp.o
CFLAGS = -O3
LDFLAGS = ${CFLAGS}

M_ARCH = $(shell uname -m)

# for Device Code
CUDA_PATH = /usr/local/cuda
CUDA_SDK_PATH = ${HOME}/NVIDIA_GPU_Computing_SDK/C
ifeq ($(M_ARCH), x86_64)
LDFLAGS += -L${CUDA_PATH}/lib64
else
LDFLAGS += -L${CUDA_PATH}/lib
endif
LDFLAGS += -L${CUDA_SDK_PATH}/lib
LDFLAGS += -lcudart -lcutil
NFLAG += ${CFLAGS} -I${CUDA_SDK_PATH}/common/inc
# NFLAG += -arch sm_13
NFLAG += -maxrregcount 32

all : ${PROG}

${PROG} : ${OBJS}
	${CC} -o $@ ${OBJS} ${LDFLAGS}

%.o : %.cu
	${NVCC} -c ${NFLAG} $<

clean :
	${RM} -f ${PROG} ${OBJS} *.o core *.bmp *.linkinfo

etc.o : etc.cu ido.h
libbmp.o : libbmp.cu
main.o : main.cu ido.h
diffusion2d.o : diffusion2d.cu ido.h

bmp:
	rm -f *.bmp
