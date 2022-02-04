CFLAGS = -O2 -Wall
CC = gcc
CUFLAGS = -O2 -g --ptxas-options=-v #-gencode arch=compute_35,code=compute_35
NVCC = nvcc

all: crypto

crypto: timer.o crypto.o
	$(NVCC) $(CUFLAGS) -o crypto timer.o crypto.o

clean:
	rm -f *.o crypto

%.o: %.cc
	$(CC) $(CFLAGS) -o $@ -c $<

%.o: %.cu
	$(NVCC) $(CUFLAGS) -o $@ -c $<
