SRC = neutron-opencl-gpu.cpp
EXEC=../neutron-opencl-gpu
LDIRS=/usr/lib/x86_64-linux-gnu


CC = g++
CFLAGS = -O3 -lOpenCL
LIBS = -lm

all: ${EXEC}



${EXEC}: $(SRC)
	$(CC) $(CFLAGS) $< -o $@ $(OBJECTS) $(LIBS) 

exec :
	./../neutron-opencl-gpu
	
clean:
	rm -f *~ ${EXEC} *.o 
