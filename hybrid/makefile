SRC = neutron-hybrid.cu
EXEC = ../neutron-hybrid

CC = nvcc
#CFLAGS = -O3 -Xcompiler -fopenmp --generate-code arch=compute_35,code=sm_35 GPU1 et 2
#CFLAGS = -O3 -Xcompiler -fopenmp --generate-code arch=compute_60,code=sm_60 GPU3
CFLAGS = -O3 -Xcompiler -fopenmp --generate-code arch=compute_60,code=sm_60
LIBS = -lm

all: ${EXEC}

${EXEC} :  $(SRC)
	$(CC) $(CFLAGS) $< -o $@ $(OBJECTS) $(LIBS)
	
exec :
	./../neutron-hybrid

clean:
	rm -f ${EXEC}

