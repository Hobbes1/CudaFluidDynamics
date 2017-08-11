#ifndef FLUIDSCUDA_H
#define FLUIDSCUDA_H

#include <GL/glew.h>
#define GLFW_DLL
#include <GLFW/glfw3.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include <GL/freeglut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define rowColIdx row*simWidth+col
#define REFRESH_DELAY 10 


struct cudaGraphicsResource *cudaVBOResource;
GLuint c_vbo;
GLuint p_vbo;

unsigned int simWidth;
unsigned int simHeight;

inline 
cudaError_t checkCuda(cudaError_t result){
	if (result != cudaSuccess){
		fprintf(stderr, "CUDA Runtime Error: %sn", cudaGetErrorString(result));
		assert(result==cudaSuccess);
	}
	return result;
}

const char* vertex_shader = 
"#version 420\n"
"layout(location=0) in vec4 vp;"
"layout(location=1) in vec3 vc;"
"out vec3 color;"
"void main() {\n"
"	color = vc;"
"  	gl_Position = vec4(vec4);}";

const char* fragment_shader = 
"#version 420\n"
"layout(location=1) in vec3 color;"
"out vec4 frag_color;"
"void main() {\n"
"	frag_color = vec4(color, 1.0);}";

void display();
bool initGL(int *argc, char **argv);

void runSim(GLuint c_vbo,
			int *argc,
			char **argv,
			struct cudaGraphicsResource **vboResource, 
			unsigned int simWidth,
			unsigned int simHeight);

void runCuda(struct cudaGraphicsResource **vboResource);

#endif
