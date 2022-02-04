CFLAGS = -O2 -Wall
CC = gcc
CUFLAGS = -O2 -g --ptxas-options=-v # -gencode arch=compute_35,code=compute_35
NVCC = nvcc

all: convolution

convolution: timer.o convolution.o
	$(NVCC) $(CUFLAGS) -o convolution timer.o convolution.o

clean:
	rm -f *.o convolution

%.o: %.cc
	$(CC) $(CFLAGS) -o $@ -c $<

%.o: %.cu
	$(NVCC) $(CUFLAGS) -o $@ -c $<
