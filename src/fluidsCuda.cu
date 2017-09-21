#include "fluidsCuda.h"

			/* Kernel to operate on color data by quadIdx, 
			 * might become null as color will be represented
			 * by velocity data directly in the future */

__global__ void
velToColor(float3 *colors,
		   float3 *colorMap,
		   float2 *__restrict__ newVel,
		   dim3 blocks,
		   unsigned int simWidth,
		   unsigned int simHeight)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	int quadIdx = x + simWidth*y;
	float magVel = sqrt(newVel[quadIdx].x * newVel[quadIdx].x + newVel[quadIdx].y * newVel[quadIdx].y);
	int map10_256 = (int)(magVel/7.0 * 256);
	if(map10_256 > 256) { map10_256 = 256; }


	for(int i = 0; i < 4; i++){
		colors[4*quadIdx+i] = colorMap[map10_256];
	}

	/*
	for(int i = 0; i < 4; i++){
		if (newVel[quadIdx].x < 0){
			colors[4*quadIdx+i].x = 1.0;
		}
	}*/
}

			/* Obstruct method, simply zero's velocities 
			 * at chosen object locations. This is probably 
			 * not what I want in the end as their should be
			 * some pressure conditions at the edge or something TODO */

__global__ void
Obstruct(int *__restrict__ obstructed,
		 float2 *__restrict__ oldVel)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	int obstruct = obstructed[x];
	oldVel[obstruct] = make_float2(0.0, 0.0);
}

			/* Diffusion method, uses iterative jacobi method 
			 * to approximate solutions to poisson's eqn in 
			 * diffusion. Should be called for a number of iterations */

__global__ void
Diffuse(float2 *__restrict__ positions,
		float2 *__restrict__ oldVel,
		float2 *__restrict__ newVel,
		float dt,
		float dr,
		float viscosity,
		unsigned int simWidth,
		unsigned int simHeight)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	float2 Vel = oldVel[y*simWidth + x];
	float2 TVel;
	float2 LVel;
	float2 BVel;
	float2 RVel;
	float alpha = dr * dr / (viscosity * dt);

	if (x!=0 && y!=0 && x!=simWidth-1 && y!=simHeight-1)
	{
		TVel = oldVel[(y-1)*simWidth + x];
		LVel = oldVel[(y*simWidth) + x - 1];
		BVel = oldVel[(y+1)*simWidth + x];
		RVel = oldVel[(y*simWidth) + x + 1];

		newVel[y*simWidth + x] = JacobiInstance(TVel, LVel, 
												BVel, RVel,
												alpha, Vel);
	}
}

			/* Advection method, utilizes backtracing to update
			 * velocities at each point on the lattice. Some 
			 * extra consideration when backtracing goes beyond
			 * simulation boundaries */

	#define TLTracedPosIdx (y-dQuadsY-1+yOff)*simWidth + (x-dQuadsX+xOff)
	#define TRTracedPosIdx (y-dQuadsY-1+yOff)*simWidth + (x-dQuadsX+1+xOff)
	#define BLTracedPosIdx (y-dQuadsY+yOff)*simWidth + (x-dQuadsX+xOff)
	#define BRTracedPosIdx (y-dQuadsY+yOff)*simWidth + (x-dQuadsX+1+xOff)

__device__ __inline__ bool
checkPosIdx (int posIdx,
			 unsigned int simWidth,
			 unsigned int simHeight)
{
	if(posIdx > simWidth*simHeight-1 || posIdx < 0){
		printf("Went out of bounds to: %i \n", posIdx);
		return false;
	}
	return true;
}

__global__ void 
Advect(float2 *__restrict__ positions,
	   float2 *__restrict__ oldVel, 
	   float2 *__restrict__ newVel,
	   float dt,
	   float dr,
	   float4 boundaries,
	   unsigned int simWidth,
	   unsigned int simHeight,
	   unsigned int testX,
	   unsigned int testY,
	   bool test)
{
		// actual realPos index
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
		// offsets to determine neighbors of interpolation with dependance on direction
	int xOff;
	int yOff;
	/*
	if(x==simWidth/4 && y==simWidth/2){
		newVel[y*simWidth+x] = make_float2(0.0, 0.0);
		return;
	}*/
	if(y*simWidth+x >= simWidth*simHeight){
		printf("Tried to access lattice position outside of memory \n");
		return;}
	
	float2 tracedPos;

	float dx = oldVel[y*simWidth+x].x * dt;
	float dy = oldVel[y*simWidth+x].y * dt;
	if (dx >= 0){ xOff = -1; } else xOff = 0;
	if (dy >= 0){ yOff = 1;} else yOff = 0;

	if(y==testY && x==testX && test==true){
		printf("dx : %f  - dy : %f\n", dx, dy);
	}

	tracedPos.x = positions[y*simWidth+x].x - oldVel[y*simWidth+x].x * dt;
	tracedPos.y = positions[y*simWidth+x].y - oldVel[y*simWidth+x].y * dt;
	if(y==testY && x==testX && test==true){
		printf("tracedPos.x : %f  - tracedPos.y : %f\n", tracedPos.x, tracedPos.y);
	}
	if(x==testX && y==testY && test==true){
		//printf("happeneddddddd");
	}

			// Top and Bottom held to zero as boundary condition
	
	if(y==0 || y==simHeight-1)
	{
		oldVel[y*simWidth+x] = make_float2(0.0, 0.0);
		return;
	}

				// change in realQuad position

	unsigned int dQuadsX = floor(dx / dr);
	unsigned int dQuadsY = floor(dy / dr);
	if(y==testY && x==testX && test==true){
		printf("dquadsX : %f  - dquadsY : %f\n", dQuadsX, dQuadsY);
	}
	if(y==testY && x==testX && test==true){
		printf("xOff : %i  - yOff : %f\n", xOff, yOff);
		printf("corners: %i %i %i %i\n", TLTracedPosIdx, TRTracedPosIdx, BLTracedPosIdx, BRTracedPosIdx);	
	}


			// is tracedPos within simulation boundaries

	if(tracedPos.x > boundaries.x && 
	   tracedPos.x < boundaries.z &&
	   tracedPos.y > (boundaries.y+dr) &&
	   tracedPos.y < boundaries.w) {
	   	if( BRTracedPosIdx >= simWidth*simHeight ){
			printf("Traced to a quadIdx that was out of bounds: %i \n", 
				  (y-dQuadsY+1)*simWidth+(x-dQuadsX));
			return;
		}

		if(y==testY && x==testX && test==true){
			printf("Tracing from: %f, %f \n", positions[y*simWidth+x].x, positions[y*simWidth+x].y);
			printf("Got traced Position: %f, %f \n", tracedPos.x, tracedPos.y);
			printf("With original velocities: %f, %f \n", oldVel[y*simWidth+x].x, oldVel[y*simWidth+x].y);
			printf("dQuads x and y: %i, %i \n", dQuadsX, dQuadsY);
		}

		float2 TLPos = positions[ TLTracedPosIdx ];
		float2 TRPos = positions[ TRTracedPosIdx ];
		float2 BLPos = positions[ BLTracedPosIdx ];
		float2 BRPos = positions[ BRTracedPosIdx];
		if(y==testY && x==testX && test==true){
			printf("Interpolating between velocities at positions: \n %f, %f \n %f, %f \n %f, %f \n %f, %f \n", TLPos.x, TLPos.y, BLPos.x, BLPos.y, BRPos.x, BRPos.y, TRPos.x, TRPos.y);
		}

		float2 TLVel = oldVel[ TLTracedPosIdx ];
		float2 TRVel = oldVel[ TRTracedPosIdx ];
		float2 BLVel = oldVel[ BLTracedPosIdx ];
		float2 BRVel = oldVel[ BRTracedPosIdx ];
		if(y==testY && x==testX && test==true){
			printf("And Velocities: \n %f, %f \n %f, %f \n %f, %f \n %f, %f \n", TLVel.x, TLVel.y, BLVel.x, BLVel.y, BRVel.x, BRVel.y, TRVel.x, TRVel.y);
		}

		if(tracedPos.y == TLPos.y){
			newVel[y*simWidth+x] = LinInterp(tracedPos,
											 TLVel, TLPos, 
											 TRVel, TRPos,
											 dr);
			
			if(y==testY && x==testX && test==true)
				printf("Velocity became from LININTERP: %f %f \n\n", newVel[y*simWidth+x].x, newVel[y*simWidth+x].y);
			

			return;
		}

		newVel[y*simWidth+x] = BiLinInterp(tracedPos, 
										   TLVel, TLPos,
										   BLVel, BLPos,
										   BRVel, BRPos,
										   TRVel, TRPos,
										   dr);
		if(y==testY && x==testX && test==true)
			printf("Velocity became from BILININTERP: %f %f \n\n", newVel[y*simWidth+x].x, newVel[y*simWidth+x].y);

		return;
		/*
		if(y*simWidth+x == simWidth*simHeight-1)
			printf("Final lattice point is doing things \n\n");
		*/
	}
		// Traced Position beyond LEFT hand boundary (x)

	if(tracedPos.x < boundaries.x &&
	   tracedPos.x < boundaries.z)
	{
		newVel[y*simWidth+x] = oldVel[y*simWidth+x];
		//newVel[y*simWidth+x].x = 3.0;
		//newVel[y*simWidth+x].y = 0.0;
		return;
	}

		// Traced Position beyond RIGHT hand boundary (z)

	if(tracedPos.x > boundaries.x && 
	   tracedPos.x > boundaries.z)
	{
		//newVel[y*simWidth+x] = oldVel[y*simWidth+x];
		
		newVel[y*simWidth+x].x = -5.0;
		newVel[y*simWidth+x].y = 0.0;
		

		return;
	}

		// Traced Position beyond TOP boundary (w)
	
	if(tracedPos.y > boundaries.y && 
	   tracedPos.y >= boundaries.w && 
	   tracedPos.x > boundaries.x &&
	   tracedPos.x < boundaries.z)
	{

		//printf("top boundary indexes : %i %i \n",  TLTracedPosIdx, TRTracedPosIdx);
		if(checkPosIdx(TLTracedPosIdx, simWidth, simHeight) && 
		   checkPosIdx(TRTracedPosIdx, simWidth, simHeight)){

		float2 LPos = positions[ TLTracedPosIdx ];
		float2 RPos = positions[ TRTracedPosIdx ];

		float2 LVel = oldVel[ TLTracedPosIdx ];
		float2 RVel = oldVel[ TRTracedPosIdx ];

		newVel[y*simWidth+x] = LinInterp(tracedPos, 	
										 LVel, LPos,
										 RVel, RPos,
										 dr);
		}
		return;
	}

		// Traced Position beyond BOTTOM boundary (y)
	
	if(tracedPos.y < boundaries.w &&
	   tracedPos.y <= boundaries.y && 
	   tracedPos.x > boundaries.x &&
	   tracedPos.x < boundaries.z)
	{
		
		//printf("bottom boundary indexes : %i %i \n",  TLTracedPosIdx, TRTracedPosIdx);
		if(checkPosIdx(TLTracedPosIdx, simWidth, simHeight) && 
		   checkPosIdx(TRTracedPosIdx, simWidth, simHeight)){

		float2 LPos = positions[ TLTracedPosIdx ];
		float2 RPos = positions[ TRTracedPosIdx ];

		float2 LVel = oldVel[ TLTracedPosIdx ];
		float2 RVel = oldVel[ TRTracedPosIdx ];

		newVel[y*simWidth+x] = LinInterp(tracedPos, 	
										 LVel, LPos,
										 RVel, RPos,
										 dr);
		}
		return;
	}
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

			/* Didn't want to write out all these multiple entries
			 * for float2 calculations every time */

__device__ float2
JacobiInstance(float2 Top, 
			   float2 Left,
			   float2 Bot,
			   float2 Right,
			   float Alpha,
			   float2 Val)
{
	float2 res;
	res.x = (Top.x + Left.x + Bot.x + Right.x + Alpha * Val.x) / (4 + Alpha);
	res.y = (Top.y + Left.y + Bot.y + Right.y + Alpha * Val.y) / (4 + Alpha);
	return res;
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
	TopInterp.x = (TRPos.x - pos.x)/(TRPos.x - TLPos.x)*TLVel.x + 
				  (pos.x - TLPos.x)/(TRPos.x - TLPos.x)*TRVel.x;

	BotInterp.x = (BRPos.x - pos.x)/(BRPos.x - BLPos.x)*BLVel.x + 
				  (pos.x - BLPos.x)/(BRPos.x - BLPos.x)*BRVel.x;
	
	TopInterp.y = (TRPos.x - pos.x)/(TRPos.x - TLPos.x)*TLVel.y + 
				  (pos.x - TLPos.x)/(TRPos.x - TLPos.x)*TRVel.y;

	BotInterp.y = (BRPos.x - pos.x)/(BRPos.x - BLPos.x)*BLVel.y + 
				  (pos.x - BLPos.x)/(BRPos.x - BLPos.x)*BRVel.y;
	
	float2 ResInterp;
	ResInterp.x = (TLPos.y - pos.y)/(TLPos.y - BRPos.y)*TopInterp.x +
				  (pos.y - BRPos.y)/(TLPos.y - BRPos.y)*BotInterp.x;

	ResInterp.y = (TLPos.y - pos.y)/(TLPos.y - BRPos.y)*TopInterp.y + 
				  (pos.y - BRPos.y)/(TLPos.y - BRPos.y)*BotInterp.y;
	
	return ResInterp;
}

			/* Linear interpolation between velocities along an edge, 
			 * between two points. Used for top/bottom edge cases where
			 * otherwise bilinear interpolation would want to read outside
			 * of data bounds */

__device__ float2
LinInterp(float2 pos,
			 float2 LVel, float2 LPos,
			 float2 RVel, float2 RPos,
			 float dr)
{
	float2 interp;
	interp.x = (RPos.x - pos.x)/(RPos.x - LPos.x)*LVel.x + 
			   (pos.x - LPos.x)/(RPos.x - LPos.x)*RVel.x;

	interp.y = (RPos.x - pos.x)/(RPos.x - LPos.x)*LVel.y + 
			   (pos.x - LPos.x)/(RPos.x - LPos.x)*RVel.y;

	return interp;
}

void runCuda(struct cudaGraphicsResource **vboResource,
			 int *obstructed,
			 float3 *colorMap,
			 float2 *pos,
			 float2 *oldVel,
			 float2 *newVel,
			 float4 boundaries,
			 float dt,
			 float dr,
			 dim3 tpbColor,
			 dim3 tpbLattice,
			 dim3 blocks,
			 unsigned int simWidth,
			 unsigned int simHeight, 
			 unsigned int testX, unsigned int testY, bool test)
{
	float3 *devPtr;
	checkCuda(cudaGraphicsMapResources(1, vboResource, 0));
	size_t numBytes;
	checkCuda(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &numBytes,
												   *vboResource));

	Obstruct<<<1, 16>>>(obstructed, oldVel);
	
	Advect<<< blocks, tpbLattice >>>(pos, oldVel, newVel, 
									 dt, dr, boundaries,
									 simWidth, simHeight, testX, testY, test);
	checkCuda(cudaPeekAtLastError());
	checkCuda(cudaDeviceSynchronize());

	updateVel<<< blocks, tpbLattice >>>(oldVel, newVel, simWidth);
	checkCuda(cudaDeviceSynchronize());

	
	float viscosity = 1.48e-5;
	for (int i = 0; i < 40; i++){
		Diffuse<<< blocks, tpbLattice >>> (pos, oldVel, newVel, 
										   dt, dr, viscosity,
										   simWidth, simHeight);
		checkCuda(cudaPeekAtLastError());
		updateVel<<< blocks, tpbLattice >>>(oldVel, newVel, simWidth);
		//std::cout<<"	Running Jacobi Diffusion: "<<i<<std::endl;
	}



	updateVel<<< blocks, tpbLattice >>>(oldVel, newVel, simWidth);

	velToColor<<< blocks, tpbLattice >>>(devPtr, colorMap, oldVel, blocks, simWidth, simHeight);
	checkCuda(cudaDeviceSynchronize());

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
						  dim3 &tpbColor,
						  dim3 &tpbLattice,
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
			
	tpbColor.x = simWidth*4/xBlocks;
	tpbColor.y = simHeight/yBlocks;
	tpbLattice.x = simWidth/xBlocks;
	tpbLattice.y = simWidth/yBlocks;
	blocks.x = xBlocks;
	blocks.y = yBlocks;
	std::cout<<"	Calling kernels with:"<<std::endl
			 <<"	ThreadsPerBlock: ["<<tpbLattice.x<<", "<<tpbLattice.y<<"]"<<std::endl
			 <<"	On a Grid of: ["<<blocks.x<<"x"<<blocks.y<<"] Blocks"<<std::endl;
}


