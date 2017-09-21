#ifndef FLUIDSCUDA_H
#define FLUIDSCUDA_H

#include <GL/glew.h>
#define GLFW_DLL
#include <GLFW/glfw3.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <assert.h>
#include <fstream>
#include <sstream>

#include <GL/freeglut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define rowColIdx row*simWidth+col
#define REFRESH_DELAY 10 

#define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t result, const char *file, int line, bool abort=true)
{
	if(result != cudaSuccess)
	{
		fprintf(stderr, "GPUAssert: %s %s %d\n", cudaGetErrorString(result), file, line);
		if (abort) exit(result);
	}
}

inline 
cudaError_t checkCuda(cudaError_t result){
	if (result != cudaSuccess){
		fprintf(stderr, "CUDA Runtime Error: %sn", cudaGetErrorString(result));
		assert(result==cudaSuccess);
	}
	return result;
}

const char *vertexShaderText = 
"#version 450 \n"
"layout(location = 0) in vec3 vertexPosition;"
"layout(location = 1) in vec3 vertexColor;"
"out vec3 color;"
"void main() {"
"	color = vertexColor;"
"	gl_Position = vec4(vertexPosition, 1.0);"
"}";

const char *fragmentShaderText = 
"#version 450\n"
"in vec3 color;"
"out vec4 fragmentColor;"
"void main() {"
"	fragmentColor = vec4(color, 1.0);"
"}";


__global__ void
Obstruct(int *__restrict__ obstructed,
		 float2 *__restrict__ oldVel);

__global__ void 
velToColor(float3 *__restrict__ colors, 
		   float3 *colorMap,
		   float2 *__restrict__ newVel,
		   dim3 blocks,
		   unsigned int simWidth,
		   unsigned int simHeight);

void runSim(GLuint c_vbo,
			int *argc,
			char **argv,
			struct cudaGraphicsResource **vboResource, 
			unsigned int simWidth,
			unsigned int simHeight);

void runCuda(struct cudaGraphicsResource **vboResource,
			 int *obstructed,
			 float3 *colorMap,
			 float2 *devPositions,
			 float2 *devVelocities,
			 float2 *devVelocities2,
			 float4 boundaries,
			 float dt,
			 float dr,
			 dim3 tpbColor,
			 dim3 tpbLattice,
			 dim3 blocks,
			 unsigned int simWidth,
			 unsigned int simHeight,
			 unsigned int testX,
			 unsigned int testY,
			 bool test);

__global__ void 
updateVel(float2 *__restrict__ oldVel,
		  float2 *__restrict__ newVel,
		  unsigned int simWidth);

__device__ float2
BiLinInterp(float2 pos,
 		 	 float2 TLVel, float2 TLPos,
			 float2 BLVel, float2 BLPos,
			 float2 BRVel, float2 BRPos,
			 float2 TRVel, float2 TRPos,
			 float dr);

__device__ float2
LinInterp(float2 pos,
			 float2 LVel, float2 LPos,
			 float2 RVel, float2 RPos,
			 float dr);

__device__ float2
JacobiInstance(float2 Top, 
			   float2 Left,
			   float2 Bot,
			   float2 Right,
			   float Alpha,
			   float2 Val);




void glInitShaders(const char *vertexShaderText,
			 	   const char *fragmentShaderText,
				   GLuint shaderProgram);

void initThreadDimensions(unsigned int simWidth,
						  unsigned int simHeight,
						  dim3 &tpb,
						  dim3 &blocks);
#endif
