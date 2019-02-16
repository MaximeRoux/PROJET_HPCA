SRC = neutron-omp.c
EXEC = ../neutron-omp

CC = gcc
CFLAGS = -Wall -O3 -fopenmp
LIBS = -lm

all: ${EXEC}

${EXEC} : $(SRC)
	$(CC) $(CFLAGS) $< -o $@ $(OBJECTS) $(LIBS)
	
exec :
	./../neutron-omp


clean:
	rm -f ${EXEC}

