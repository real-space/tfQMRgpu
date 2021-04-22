EXECUTABLE = bench_tfqmrgpu.exe
all: ${EXECUTABLE}

# OBJ = test_tfqmrgpu.o bench_tfqmrgpu.o tfqmrgpu.o
OBJ = bench_tfqmrgpu.o tfqmrgpu.o

.PHONY: clean allclean
clean:
	-rm -f *.o *~ *.d *.mod

allclean: clean
	-rm -f ${EXECUTABLE}

CC     = gcc
CXX    = g++
CU     = nvcc

##MPI
# CC  = mpicc
# CXX = mpic++

SRC ?= .
INCLUDES=
INCLUDES += -I../include
# INCLUDES += -I/usr/local/cuda/include

EXTFLAGS ?=
EXTFLAGS += -DDEBUG
EXTFLAGS += -DDEBUGGPU

ERRFLAGS = -Werror -Wall -Wfatal-errors -Wno-sign-compare

FEAFLAGS=
FEAFLAGS += -DHAS_NO_CUDA
# FEAFLAGS += -fopenmp
FEAFLAGS += -fno-omit-frame-pointer

# normal mode
OPTFLAGS = -Ofast -mtune=power8 -mcpu=power8 -mpower8-fusion -mpower8-vector -mvsx -maltivec -mdirect-move -fstrict-aliasing -fomit-frame-pointer -fno-schedule-insns
CUOPTFLAGS=
CUOPTFLAGS += -O3 --use_fast_math
CUOPTFLAGS += -DDEBUGGPU
CUOPTFLAGS += -DDEBUG

## verbose assembler
# CUOPTFLAGS +=  -Xptxas -v

# debug mode
# OPTFLAGS  = -O0
OPTFLAGS  = -g
# CUOPTFLAGS += -g -O0 -G # should go here, but is not working due to bug (error in ptx assembly...)

# for profiling
CUOPTFLAGS += -lineinfo
# FEAFLAGS += -DUSE_NVTX

FLAGS = ${ERRFLAGS} ${FEAFLAGS} ${OPTFLAGS} ${EXTFLAGS}

GENCODE_FLAGS =
# GENCODE_FLAGS += -gencode arch=compute_35,code=sm_35
# GENCODE_FLAGS += -gencode arch=compute_37,code=sm_37
# GENCODE_FLAGS += -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61
GENCODE_FLAGS += -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70 ## V100
# GENCODE_FLAGS += -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80 ## A100

CCFLAGS  = -std=gnu99 ${FLAGS}
CXXFLAGS = -std=c++11 ${FLAGS}
CUFLAGS  = -std=c++11  ${GENCODE_FLAGS} ${CUOPTFLAGS} $(patsubst %,-Xcompiler %,${FLAGS})
#CUFLAGS  = ${CXXFLAGS}
# CUFLAGS += -nocudainc -nocudalib

LDFLAGS =
# LDFLAGS += -lnvToolsExt
LDFLAGS += -lm -g -rdynamic 
# LDFLAGS += -nocudalib

# HOSTNAME := $(shell command -v hostname)
HOSTNAME = "zam054"
ifeq ($(HOSTNAME), "zam054")
# 	LDFLAGS += -L/usr/local/zam/CUDA/cuda-9.1.85/lib64
# 	LDFLAGS += -L/usr/local/zam/CUDA/cuda-10.1.168/lib64
else
# 	LDFLAGS += -L/usr/local/cuda/lib64 
endif
### random numbers created on the GPU
# LDFLAGS += -lcudart -lcurand

DEP=$(OBJ:%.o=%.d)

${EXECUTABLE}: ${OBJ}
	${CXX} ${CXXFLAGS} $^ ${LDFLAGS} -o $@

%.o:%.c
	${CC} ${CCFLAGS} ${INCLUDES} -c $<
%.o:%.cxx
	${CXX} ${CXXFLAGS} ${INCLUDES} -c $<
%.o:%.cu
	${CU} ${CUFLAGS} ${INCLUDES} -c $<

%.d:%.c
	${CC} ${CFLAGS} ${INCLUDES} -MM $^ -MF $@
%.d:%.cu
	${CXX} -x c++ ${CXXFLAGS} ${INCLUDES} -MM $^ -MF $@
%.d:%.cxx
	${CXX} ${CXXFLAGS} ${INCLUDES} -MM $^ -MF $@

-include ${DEP}
