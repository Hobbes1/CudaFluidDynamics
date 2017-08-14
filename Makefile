CU_FILES := $(wildcard src/*.cu)
OBJ_FILES := $(addprefix obj/,$(notdir $(CU_FILES:.cu=.o)))

LIBS = -lglfw -lGL -lGLEW
NVCC = nvcc  
NVFLAGS = -arch=sm_30
CC = g++
CFLAGS = -std=c++11



gltest: $(OBJ_FILES) 
	$(NVCC) $(NVFLAGS) -o gltest main.o $(LIBS)

obj/%.o: src/%.cu
	$(NVCC) $(NVFLAGS) -c $< $(LIBS)



clean: 
	rm -f *.o




