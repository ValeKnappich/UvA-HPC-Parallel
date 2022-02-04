NVCC		= nvcc
CC			= g++
CU_FLAGS	= -O2 -g --ptxas-options=-v -gencode arch=compute_35,code=compute_35
CC_FLAGS	= -O2 -g -m64 -Wall

all: 		vector-add vector-add-seq

%.o:		%.cu
		$(NVCC) $(CU_FLAGS) -c $< -o $@

%.o:		%.cc
		$(CC) $(CC_FLAGS) -c $< -o $@

vector-add:	vector-add.o
		$(NVCC) $^ -o $@

vector-add-seq: vector-add-seq.o
		$(CC) $^ -o $@

clean:
		rm -f *.o vector-add vector-add-seq
