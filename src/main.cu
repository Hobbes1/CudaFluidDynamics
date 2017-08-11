#include "fluidsCuda.h"
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
	simWidth = 4;
	simHeight = 4;

	// start GL context and O/S window using the GLFW helper library
	if (!glfwInit()) {
		fprintf(stderr, "ERROR: could not start GLFW3\n");
		return 1;
	} 

	GLFWwindow* window = glfwCreateWindow(1600, 1000, "Cuda Fluid Dynamics", NULL, NULL);
	if (!window) {
		fprintf(stderr, "ERROR: could not open window with GLFW3\n");
		glfwTerminate();
		return 1;
	}
	glfwMakeContextCurrent(window);

	glewExperimental = GL_TRUE;
	glewInit();

	const GLubyte* renderer = glGetString(GL_RENDERER); 
	const GLubyte* version = glGetString(GL_VERSION); 
	printf("Renderer: %s\n", renderer);
	printf("OpenGL version supported %s\n", version);

			///*** Post GL Initializatiuon ***///

	int indexCount = 4 + 										// corners
					 (simWidth - 2) * 4 + 						// t/b sides
					 (simHeight - 2) * 4 +						// r/l sides
					 (simWidth - 2)* (simHeight - 2) * 4;		// inner

	float3 *points = (float3*)malloc(indexCount*sizeof(float3));
	float3 *colors = (float3*)malloc(indexCount*sizeof(float3));
	cout << indexCount << " indexCount " << endl;

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

	cout << "colors: " << indexCount<<endl;
	quadNum = 0;
	for (int i = 0; i < indexCount; i++){
		colors[i].x = 1.0f;
		colors[i].y = (float)quadNum/(simWidth*simHeight); cout<<colors[i].y;
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
	
	while(!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glUseProgram(shaderProgram);

		glBindVertexArray(vertexArray);

		glDrawArrays(GL_QUADS, 0, indexCount);

		glfwSwapBuffers(window);
  		glfwPollEvents();

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
