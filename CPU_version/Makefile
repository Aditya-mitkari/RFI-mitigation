SHELL = /bin/sh

#====================================================
# Set executable, sources and objects
#====================================================
EXE = rfi.ex
SRC = *.cpp
CPP_OBJ = rfi.o


#====================================================
# Set compiler
#====================================================
COMP=pgi
#COMP=intel

#=========================================================
# CUDA_PATH on summit, should be changed for other machines
#=========================================================

OPENACC=no

ifeq ($(PRODUCTION),y)
	DEFINE += -DPRODUCTION_VERSION
endif

#====================================================
# Flag to set NVTX range marker
#====================================================
ifeq ($(USE_NVTX),y)
	DEFINE += -DUSE_NVTX
endif

CXXFLAGS = -O3 -std=c++11

#====================================================
# PGI compiler for OpenACC and CUDA compiler
#====================================================
ifeq ($(COMP),pgi)
CXX=pgc++
CXX_CU=nvcc

#====================================================
# PGI compiler flags
#====================================================
CXXFLAGS = -fast -lm
CXXFLAGS += -I /usr/local/cuda-10.1/include

ifeq ($(OPENACC),y)
CXXFLAGS += -acc
CXXFLAGS += -ta=tesla:cc60
CXXFLAGS += -Minline=rand #-Minfo
CXXFLAGS += -Mcuda -Mcudalib=curand
#CXXFLAGS += -Xptxas -rdc=true
LDFLAGS = -acc -Mcuda -ta=tesla:cc60
LDFLAGS += -lcurand
CXXFLAGS += -Minfo=accel
endif
endif
LDFLAGS += -L /usr/local/cuda-10.1/lib64/

ifeq ($(USE_NVTX),y)
LDFLAGS += -lnvToolsExt
endif

ifeq ($(COMP),intel)
CXX=icc
CXXFLAGS += -qopt-report=5
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
