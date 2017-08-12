#include "fluidsCuda.h"

/*
void initCudaVBO(GLuint *vbo, 
				 struct cudaGraphicsResource **vboRes, 
			 	 unsigned int vboResFlags, 
				 unsigned int simHeight,
				 unsigned int simWidth)
{
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	unsigned int size = simWidth * simHeight * 3 * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	checkCuda(cudaGraphicsGLRegisterBuffer(vboRes, *vbo, vboResFlags));
}*/

__global__ void
testKernel(float3 *colors,
		   float time,
		   int xBlocks,
		   unsigned int simWidth,
		   unsigned int simHeight)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	
		// bundle color ops to quads

	int quadIdx = (x + y*blockDim.x*xBlocks)/4;
	int corner = (x + y*blockDim.x)%4;

		// Positions in quad space 

	int u = (simWidth-1) - (quadIdx % (simWidth-1));
	int v = (quadIdx / (simHeight - 1));


	colors[4*quadIdx+corner] = make_float3(cosf(u/3.14f - time), sinf(v/3.14f + time) + sinf(u/3.14f + time), sinf(2*v/3.14f + time));
	
}

/*
void runSim(GLuint c_vbo,
			int *argc,
			char **argv,
			struct cudaGraphicsResource **vboResource, 
			unsigned int simWidth,
			unsigned int simHeight)
{

	initGL(argc, argv);

	glutDisplayFunc(display);

	//runCuda(vboResource);

	glutMainLoop();
}*/

void runCuda(struct cudaGraphicsResource **vboResource,
			 float time,
			 unsigned int simWidth,
			 unsigned int simHeight)
{
	float3 *devPtr;
	checkCuda(cudaGraphicsMapResources(1, vboResource, 0));
	size_t numBytes;
	checkCuda(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &numBytes,
												   *vboResource));
	
			/* Computing downwards by rows of 2d space, select
			 * a block size depending on how many threads. Aiming 
			 * for 512 block size, i.e. if simulation size 
			 * requires 4096 threads, calc is done in 8 blocks */

	int xBlocks;
	int yBlocks;
	int numThreads = (simWidth-1)*4 * (simHeight-1);	
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
			return;
	}
			
	dim3 tpb((simWidth-1)*4/xBlocks, (simHeight-1)/yBlocks);
	dim3 blocks(xBlocks, yBlocks);
	/*
	std::cout<<"	Calling kernel with:"<<std::endl
			 <<"	ThreadsPerBlock: ["<<tpb.x<<", "<<tpb.y<<std::endl
			 <<"	On a Grid of: ["<<blocks.x<<"x"<<blocks.y<<" Blocks"<<std::endl;
	*/
	testKernel<<< blocks, tpb >>>(devPtr, time, xBlocks, simWidth, simHeight);
	//std::cout<<cudaGetErrorString(cudaGetLastError())<<std::endl;

	checkCuda(cudaGraphicsUnmapResources(1, vboResource, 0));
}


	

/*
bool initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(1600, 1000);
    glutCreateWindow("Cuda GL Test");
    glutDisplayFunc(display);

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, 1600, 1000);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)1600 / (GLfloat)1000, 0.1, 10.0);

    return true;
}*/
/*
void display()
{
    // run CUDA kernel to generate vertex positions
	//runCuda(&cudaVBOResource);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, c_vbo);
    glVertexPointer(3, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glDrawArrays(GL_QUADS, 0, simWidth * simHeight);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();
}*/

