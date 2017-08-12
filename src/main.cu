#include "fluidsCuda.h"
#include "fluidsCuda.cu"
#include <math.h>
using namespace std;

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

int main(int argc, char *argv[])
{
	simWidth = 257;
	simHeight = 257;

	// start GL context and O/S window using the GLFW helper library
	if (!glfwInit()) {
		fprintf(stderr, "ERROR: could not start GLFW3\n");
		return 1;
	} 

	GLFWwindow* window = glfwCreateWindow(1920/2, 1080/2, "Cuda Fluid Dynamics", NULL, NULL);
	glfwSetWindowPos(window, 1920/2, 0);
	if (!window) {
		fprintf(stderr, "ERROR: could not open window with GLFW3\n");
		glfwTerminate();
		return 1;
	}
	glfwMakeContextCurrent(window);

	glewExperimental = GL_TRUE;
	glewInit();

	//const GLubyte* renderer = glGetString(GL_RENDERER); 
	//const GLubyte* version = glGetString(GL_VERSION); 
	//printf("Renderer: %s\n", renderer);
	//printf("OpenGL version supported %s\n", version);

			///*** Post GL Initializatiuon ***///

	int indexCount = 4 + 										// corners
					 (simWidth - 2) * 4 + 						// t/b sides
					 (simHeight - 2) * 4 +						// r/l sides
					 (simWidth - 2)* (simHeight - 2) * 4;		// inner

	float3 *points = (float3*)malloc(indexCount*sizeof(float3));
	float3 *colors = (float3*)malloc(indexCount*sizeof(float3));

	int idxRow = 0;
	int idxCol = 0;
	int quadNum = 0;
	float halfHeight = (float)1.0/simHeight;
	float halfWidth = (float)1.0/simWidth;
	for (int i = 0; i < indexCount; i++){
		if(i%4==0){ //top left of a quad
			points[i].x = 2.0*(float)idxCol/simWidth - 1.0f + halfWidth;
			points[i].y = 1.0f - 2.0*(float)idxRow/simHeight - halfHeight;
			points[i].z = 0.0f;
		}
		if(i%4==1){ //bottom left of a quad
			points[i].x = 2.0*(float)idxCol/simWidth - 1.0f + halfWidth;
			points[i].y = 1.0f - 2.0*(float)(idxRow+1)/simHeight - halfWidth;
			points[i].z = 0.0f;
		}
		if(i%4==2){ //bottom right of a quad
			points[i].x = 2.0*(float)(idxCol+1)/simWidth - 1.0f + halfWidth;
			points[i].y = 1.0f - 2.0*(float)(idxRow+1)/simHeight - halfHeight;
			points[i].z = 0.0f;
		}
		if(i%4==3){ //top right of a quad
			points[i].x = 2.0*(float)(idxCol+1)/simWidth - 1.0f + halfWidth;
			points[i].y = 1.0f - 2.0*(float)idxRow/simHeight - halfHeight;
			points[i].z = 0.0f;
			idxCol++;
		}
		if(idxCol == simWidth-1){ // row of quads done
			idxCol = 0;
			idxRow++;
		}
	}

	cout << "	Color Data: " << indexCount<<endl;
	quadNum = 0;
	for (int i = 0; i < indexCount; i++){
		colors[i].x = 0.0f;
		colors[i].y = 0.0f; 
		colors[i].z = 0.0f;
		if(i%4==0 && i!=0){
			quadNum++;
		}
	}
			///*** Create position and color vertex buffer objects ***///

		// Position Buffer Array
	GLuint pointsVBO = 0;
	glGenBuffers(1, &pointsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, pointsVBO);
	glBufferData(GL_ARRAY_BUFFER, indexCount*sizeof(float3), points, GL_DYNAMIC_DRAW);

		// Color Buffer Array
	GLuint colorsVBO = 0;
	glGenBuffers(1, &colorsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, colorsVBO);
	glBufferData(GL_ARRAY_BUFFER, indexCount*sizeof(float3), colors, GL_DYNAMIC_DRAW);
	struct cudaGraphicsResource *cudaColorResource;
	checkCuda(cudaGraphicsGLRegisterBuffer(&cudaColorResource, colorsVBO, cudaGraphicsMapFlagsNone));

		// Index Buffer Array

			///*** vao binding ***///
	GLuint vertexArray = 0;
	glGenVertexArrays(1, &vertexArray);
	glBindVertexArray(vertexArray);

	glBindBuffer(GL_ARRAY_BUFFER, pointsVBO);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glBindBuffer(GL_ARRAY_BUFFER, colorsVBO);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

			///*** Create shader program, vertex and fragment shaders, attach and link ***///

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
		cout << log;
		return 1;
	}
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderText, NULL);
	glCompileShader(fragmentShader);
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &params);
	if(GL_TRUE != params){	
		fprintf(stderr, "ERROR: GL shader idx %i did not compile\n", fragmentShader);
		return 1;
	}
	GLuint shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

			///*** Draw Loop ***///
	
	float simTime = 0.0;
	float dt = 0.1;
	cudaEvent_t start, stop; 
	float fpsTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
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
			return 0;
	}
			
	dim3 tpb((simWidth-1)*4/xBlocks, (simHeight-1)/yBlocks);
	dim3 blocks(1, yBlocks);
	std::cout<<"	Calling kernel with:"<<std::endl
			 <<"	ThreadsPerBlock: ["<<tpb.x<<", "<<tpb.y<<"]"<<std::endl
			 <<"	On a Grid of: ["<<blocks.x<<"x"<<blocks.y<<"] Blocks"<<std::endl;


	while(!glfwWindowShouldClose(window)) {

		cudaEventRecord(start, 0);
		simTime+=dt;
		runCuda(&cudaColorResource, simTime, simWidth, simHeight);
		
		


		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glUseProgram(shaderProgram);
		glBindVertexArray(vertexArray);
		glDrawArrays(GL_QUADS, 0, indexCount);

		glfwSwapBuffers(window);
  		glfwPollEvents();

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&fpsTime, start, stop);
		char title[256];
		sprintf(title, "CudaFluidDynamics: %12.2f fps", 1.0f/(fpsTime/1000.0f));
		glfwSetWindowTitle(window, title);

		

		if(glfwGetKey(window, GLFW_KEY_ESCAPE)) {
			glfwSetWindowShouldClose(window, 1);
		}
	}

	glDeleteBuffers(1, &pointsVBO);
	glDeleteBuffers(1, &colorsVBO);
	glDeleteVertexArrays(1, &vertexArray);
	free(points);
	free(colors);
	glfwTerminate();
	return 0;
}
