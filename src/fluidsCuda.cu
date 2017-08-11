#include "fluidsCuda.h"

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
}

__global__ void
testKernel(float3 *colors,
		   unsigned int simWidth,
		   unsigned int simHeight)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	float u = x / (float)simWidth;
	float v = y / (float)simHeight;

	u = u * 2.0f - 1.0f;
	v = v * 2.0f - 1.0f;

	colors[y*simWidth+x] = make_float3(u, v, 1.0f);
}

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
}

/*
void runCuda(struct cudaGraphicsResource **vboResource)
{
	float3 *devPtr;
	checkCuda(cudaGraphicsMapResources(1, vboResource, 0));
	size_t numBytes;
	checkCuda(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &numBytes,
												   *vboResource));
	
	dim3 block(8, 8, 1);
	dim3 grid(simWidth / block.x, simHeight/block.y, 1);
	testKernel<<< grid, block >>>(colors, simWidth, simHeight);

	checkCuda(cudaGraphicsUnmapResources(1, vboResource, 0));
}*/


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
}

void display()
{
    // run CUDA kernel to generate vertex positions
	//runCuda(&cudaVBOResource);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
	/*
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);
	*/

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, c_vbo);
    glVertexPointer(3, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glDrawArrays(GL_QUADS, 0, simWidth * simHeight);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();
}

