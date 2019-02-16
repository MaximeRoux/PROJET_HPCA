# nvcc premiers.cu -o premiers -O3 --generate-code arch=compute_35,code=sm_35

SRC = neutron-mpi-cuda.cu
EXEC = ../neutron-mpi-cuda
LIB= -lm -lgomp -L/usr/local/cuda/lib64 -lcuda -lcudart -lcurand -lmpi 
CC = nvcc
ARCH1 = --generate-code arch=compute_35,code=sm_35
ARCH2 = --generate-code arch=compute_60,code=sm_60
LIBS = -lm -lmpi

NVCCFLAGS = --compiler-option="-Wall -Wextra -Werror -O3"
CUDA_FLAGS = $(ARCH1) $(ARCH2) -O3
CFLAGS = -Wall -Wextra -Werror -O3

all: ${EXEC}

neutron-mpi-cuda.o : neutron-mpi-cuda.cu
	nvcc $(CUDA_FLAGS) -c $<

${EXEC} : neutron-mpi-cuda.o
	mpic++ $(CFLAGS) -o $@ $+ $(LIB) 
	
exec :
	mpirun -n 2 -hostfile hostfile --map-by node ./../neutron-mpi-cuda
clean:
	rm -f ${EXEC} *.o

 


