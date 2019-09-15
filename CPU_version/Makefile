SHELL = /bin/sh

#====================================================
# Set executable, sources and objects
#====================================================
EXE = rfi.ex
SRC = *.cpp

CPP_OBJ = rfi.o

#=========================================================
# CUDA_PATH on summit, should be changed for other machines
#=========================================================

OPENACC=no

ifeq ($(PRODUCTION),y)
	DEFINE += -DPRODUCTION_VERSION
endif



#====================================================
# PGI compiler for OpenACC and CUDA compiler
#====================================================
COMP = pgi
CXX=pgc++
CXX_CU=nvcc

#====================================================
# PGI compiler flags
#====================================================
CXXFLAGS = -O3 -fast -std=c++11 -lm

ifeq ($(OPENACC),y)
CXXFLAGS += -acc
CXXFLAGS += -ta=tesla:cc60,managed,nollvm
CXXFLAGS += -Minline=rand #-Minfo
CXXFLAGS += -Mcuda -Mcudalib=curand
CXXFLAGS += -I /usr/local/cuda-10.1/include
#CXXFLAGS += -Xptxas -rdc=true
LDFLAGS = -acc -Mcuda -ta=tesla:cc60,managed
LDFLAGS += -lcurand
CXXFLAGS += -Minfo=accel
endif

CXXFLAGS += $(DEFINE)

#====================================================
# CUDA flags needed
#====================================================
#cudaflags:= -arch=sm_70
#NVCCFLAGS = -ccbin pgc++ -O3 -std=c++11 -Wno-deprecated-gpu-targets $(cudaflags)
#NVCCFLAGS += -I$(CUDA_PATH)/samples/common/inc/
#NVCCFLAGS += -Xcompiler "-I$(CUDA_PATH)/include/ CUDA_HOME=$(CUDA_PATH) -I$(CUDA_PATH)/samples/common/inc/ "

#==========================
# Make the executable
#==========================
#
$(EXE): $(CPP_OBJ)
	$(CXX) $(CPP_OBJ) $(LDFLAGS) -o $(EXE)

#%.o: %.cu
#	$(CXX_CU) $(NVCCFLAGS) $(DEFINE) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

##==========================
#remove all objs
#==========================
clean:
	/bin/rm -f *.o $(EXE)
