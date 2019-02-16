SRC = neutron-seq.c
EXEC = ../neutron-seq

CC = gcc
CFLAGS = -Wall -O3
LIBS = -lm

all: ${EXEC}

${EXEC} : $(SRC)
	$(CC) $(CFLAGS) $< -o $@ $(OBJECTS) $(LIBS)

clean:
	rm -f ${EXEC}

