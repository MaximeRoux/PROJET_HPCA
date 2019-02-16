# nvcc premiers.cu -o premiers -O3 --generate-code arch=compute_35,code=sm_35

SRC = neutron-cuda.cu
EXEC = ../neutron-cuda

CC = nvcc
#CFLAGS = -O3 --generate-code arch=compute_35,code=sm_35 GPU1 et 2
#CFLAGS = -O3 --generate-code arch=compute_60,code=sm_60 GPU3
CFLAGS = -O3 --generate-code arch=compute_60,code=sm_60
LIBS = -lm

all: ${EXEC}

${EXEC} : $(SRC)
	$(CC) $(CFLAGS) $< -o $@ $(OBJECTS) $(LIBS)
	
exec :
	./../neutron-cuda

clean:
	rm -f ${EXEC}

