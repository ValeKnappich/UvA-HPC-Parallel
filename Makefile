EXECUTABLES=baseline omp

EXPENSIVE_JUNK += $(EXECUTABLES)

SRC = utils.c baseline.c omp.c

JUNK +=

# CFLAGS += -O3 -Wall -W --std=c11 -lm
# CXXFLAGS += -O3 -Wall -W --std=c++11 -lm -Wno-cast-function-type
# OMP_CFLAGS = $(CFLAGS) -fopenmp
# MPI_CFLAGS = $(CXXFLAGS) -lmpi
FLAGS += -O3 -Wall -W --std=c11 -lm -fopenmp

help:
	@echo "help\tShow this help text"
	@echo "all\tMake all executables"
	@echo "clean\tThrow away all files that are easy to produce again"
	@echo "empty\tThrow away all files that can be produced again"

all: $(EXECUTABLES)

clean:
	rm -rf $(JUNK)

empty:
	rm -rf $(JUNK) $(EXPENSIVE_JUNK)

baseline: baseline.c utils.c
	$(CC) $(FLAGS) -o baseline utils.c baseline.c

omp: omp.c utils.c
	$(CC) $(FLAGS) -o omp utils.c omp.c