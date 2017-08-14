CU_FILES := $(wildcard src/*.cu)
OBJ_FILES := $(addprefix obj/,$(notdir $(CU_FILES:.cu=.o)))

LIBS = -lglfw -lGL -lGLEW
NVCC = nvcc
NVFLAGS = -arch=sm_30

OBJDIR = obj



gltest: $(OBJ_FILES) 
	$(NVCC) $(NVFLAGS) -o gltest $(OBJDIR)/main.o $(LIBS)

obj/%.o: $(CU_FILES)
	$(NVCC) $(NVFLAGS) -c $< -o $@ $(LIBS)



clean: 
	rm -f *.o




