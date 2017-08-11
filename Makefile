CPP_FILES := $(wildcard *.cpp)
CU_FILES := $(wildcard src/*.cu)
CPP_OBJ_FILES := $(addprefix obj/,$(notdir $(CPP_FILES:.cpp=.o)))
CU_OBJ_FILES := $(addprefix obj/,$(notdir $(CU_FILES:.cu=.o)))

LIBS = -lglfw -lGL -lGLEW
NVCC = nvcc  
NVFLAGS = -arch=sm_30
CC = g++
CFLAGS = -std=c++11

gltest: $(CU_OBJ_FILES)
	$(NVCC) $(NVFLAGS) -o gltest main.o $(LIBS)

obj/%.o: %.cu
	$(NVCC) $(NVFLAGS) -c $< $(LIBS)




