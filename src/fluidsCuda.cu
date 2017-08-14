#include "fluidsCuda.h"

			/* Kernel to operate on color data by quadIdx, 
			 * might become null as color will be represented
			 * by velocity data directly in the future */

__global__ void
testKernel(float3 *colors,
		   float time,
		   dim3 blocks,
		   unsigned int simWidth,
		   unsigned int simHeight)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	
		// bundle color ops to quads

	int quadIdx = (x + y*blockDim.x*blocks.x)/4;
	int corner = (x + y*blockDim.x)%4;

		// Positions in quad space 

	int u = simWidth - (quadIdx % simWidth);
	int v = quadIdx / simHeight;
	colors[4*quadIdx+corner] = make_float3(cosf(u/3.14f - time), 
										   sinf(v/3.14f + time) + sinf(u/3.14f + time), 
										   sinf(2*v/3.14f + time));
}

			/* Update old velocity data to current velocity data.
			 * These update at random and so must be updated in 
			 * bulk after velocity calculations are done */

__global__ void 
updateVel(float2 *__restrict__ oldVel,
		  float2 *__restrict__ newVel,
		  unsigned int simWidth)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	oldVel[y*simWidth+x] = newVel[y*simWidth+x];
}

			/* Bilinear Interpolation of velocities at four nearest
			 * mesh points giving the expected velocity at an arbitrary
			 * point contained */

__device__ float2
BiLinInterp(float2 pos,
			float2 TLVel, float2 TLPos,
			float2 BLVel, float2 BLPos,
			float2 BRVel, float2 BRPos,
			float2 TRVel, float2 TRPos,
			float dr)
{
	float2 TopInterp;
	float2 BotInterp;
	TopInterp.x = TLVel.x + (TRVel.x - TLVel.x)*pos.x/dr;
	BotInterp.x = BLVel.x + (BRVel.x - BLVel.x)*pos.x/dr;
	TopInterp.y = TLVel.y + (TRVel.y - TLVel.y)*pos.x/dr;
	BotInterp.y = BLVel.y + (BRVel.y - BLVel.y)*pos.x/dr;

	float2 ResInterp;
	ResInterp.x = BotInterp.x + (TopInterp.x - BotInterp.x)*pos.y/dr;
	ResInterp.y = BotInterp.y + (TopInterp.y - BotInterp.y)*pos.y/dr;

	return ResInterp;
}


			/* Advection method, utilizes backtracing to update
			 * velocities at each point on the lattice. Some 
			 * extra consideration when backtracing goes beyond
			 * simulation boundaries */

__global__ void 
Advect(float2* positions,
	   float2* oldVel, 
	   float2* newVel,
	   float dt,
	   float dr,
	   float4 boundaries,
	   unsigned int simWidth,
	   unsigned int simHeight)
{
		// actual realPos index
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	float2 tracedPos;

	float dx = oldVel[x*simWidth+y].x * dr * dt;
	float dy = oldVel[x*simWidth+y].y * dr * dt;

	tracedPos.x = positions[x*simWidth+y].x - oldVel[x*simWidth+y].x * dr * dt;
	tracedPos.y = positions[x*simWidth+y].y - oldVel[x*simWidth+y].y * dr * dt;

			// change in realQuad position

	unsigned int dQuadsX = floor(dx / dr);
	unsigned int dQuadsY = floor(dy / dr);

			// is tracedPos within simulation boundaries

	if(tracedPos.x > boundaries.x && 
	   tracedPos.x < boundaries.z &&
	   tracedPos.y > boundaries.y &&
	   tracedPos.y < boundaries.w) {

		float2 TLPos = positions[(x-dQuadsX-1)*simWidth+(y+dQuadsY)];
		float2 TRPos = positions[(x-dQuadsX)*simWidth+(y+dQuadsY)];
		float2 BLPos = positions[(x-dQuadsX-1)*simWidth+(y+dQuadsY+1)];
		float2 BRPos = positions[(x-dQuadsX)*simWidth+(y+dQuadsY+1)];

		float2 TLVel = oldVel[(x-dQuadsX-1)*simWidth+(y+dQuadsY)];
		float2 TRVel = oldVel[(x-dQuadsX)*simWidth+(y+dQuadsY)];
		float2 BLVel = oldVel[(x-dQuadsX-1)*simWidth+(y+dQuadsY+1)];
		float2 BRVel = oldVel[(x-dQuadsX)*simWidth+(y+dQuadsY+1)];

		newVel[x*simWidth+y] = BiLinInterp(tracedPos, 
										   TLVel, TLPos,
										   BLVel, BLPos,
										   BRVel, BRPos,
										   TRVel, TRPos,
										   dr);
	}
			// TODO need cases for velocity sources for hitting x/y/z/w boundaries
}


void runCuda(struct cudaGraphicsResource **vboResource,
			 float dt,
			 dim3 tpb,
			 dim3 blocks,
			 unsigned int simWidth,
			 unsigned int simHeight)
{
	float3 *devPtr;
	checkCuda(cudaGraphicsMapResources(1, vboResource, 0));
	size_t numBytes;
	checkCuda(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &numBytes,
												   *vboResource));
	
	testKernel<<< blocks, tpb >>>(devPtr, dt, blocks, simWidth, simHeight);
	checkCuda(cudaGraphicsUnmapResources(1, vboResource, 0));
}

void glfwInitContext(GLFWwindow* window)
{
	if (!glfwInit()) {
		fprintf(stderr, "ERROR: could not start GLFW3\n");
		exit(1);
	} 

	window = glfwCreateWindow(1920/2, 1080/2, "Cuda Fluid Dynamics", NULL, NULL);
	glfwSetWindowPos(window, 1920/2, 0);
	if (!window) {
		fprintf(stderr, "ERROR: could not open window with GLFW3\n");
		glfwTerminate();
		exit(1);
	}
	glfwMakeContextCurrent(window);

	glewExperimental = GL_TRUE;
	glewInit();
}

void glInitShaders(const char *vertexShaderText,
			 	   const char *fragmentShaderText,
				   GLuint shaderProgram)
{
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexShaderText, NULL);
	glCompileShader(vertexShader);
	int params = -1;
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &params);
	if(GL_TRUE != params){	
		int actual_length = 0;
		char log[2048];
		fprintf(stderr, "ERROR: GL shader idx %i did not compile\n", vertexShader);
		glGetShaderInfoLog(vertexShader, 500, &actual_length, log);
		std::cout << log;
		exit(1);
	}
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderText, NULL);
	glCompileShader(fragmentShader);
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &params);
	if(GL_TRUE != params){	
		int actual_length = 0;
		char log[2048];
		fprintf(stderr, "ERROR: GL shader idx %i did not compile\n", vertexShader);
		glGetShaderInfoLog(vertexShader, 500, &actual_length, log);
		std::cout << log;
		exit(1);
	}
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);
}

void initThreadDimensions(unsigned int simWidth,
						  unsigned int simHeight,
						  dim3 &tpb,
						  dim3 &blocks)
{
	int xBlocks;
	int yBlocks;
	int numThreads = simWidth*4*simHeight;	
	switch(numThreads){
		case 1024:
			xBlocks = 1;
			yBlocks = 8;
			break;
		case 4096:
			xBlocks = 1;
			yBlocks = 8;
			break;
		case 16384:
			xBlocks = 1;
			yBlocks = 32;
			break;
		case 65536:
			xBlocks = 1;
			yBlocks = 128;
			break;
		case 262144:
			xBlocks = 2;
			yBlocks = 256;
			break;
		default:
			std::cout<<"Bad Dimensions"<<std::endl;
			exit(1);
	}
			
	tpb.x = simWidth*4/xBlocks;
	tpb.y = simHeight/yBlocks;
	blocks.x = xBlocks;
	blocks.y = yBlocks;
	std::cout<<"	Calling kernels with:"<<std::endl
			 <<"	ThreadsPerBlock: ["<<tpb.x<<", "<<tpb.y<<"]"<<std::endl
			 <<"	On a Grid of: ["<<blocks.x<<"x"<<blocks.y<<"] Blocks"<<std::endl;
}


