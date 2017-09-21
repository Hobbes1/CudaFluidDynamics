#include "fluidsCuda.h" 
#include "fluidsCuda.cu"
using namespace std;

int main(int argc, char *argv[])
{
	unsigned int simWidth = 256;	// x divisions
	unsigned int simHeight = 256;// y divisions
									// note: 6mx6m with 256x256 yields ~square inch resolution
	float xRange = 6.0; 			// meters
	float yRange = 6.0;				// meters
	float frameVel = -3.0;			// speed of object, right to left


		// testing parameters, TODO to be removed 
	unsigned int testX = simWidth - 10;
	unsigned int testY = simHeight - 10;
	bool test = false;
	int testNum = 300;

	float dt = 0.001;

	int colorIndexCount = simWidth * simHeight * 4;
	int realIndexCount = simWidth * simHeight;
	cout <<"	frameVel will cover "<<frameVel*dt/(xRange/simWidth)<<" quads" <<endl;

			/////////*** Quad Vertices and Color data, 4x as large as real data ***/////////

	float3 *quadPoints = (float3*)malloc(colorIndexCount*sizeof(float3));
	float3 *colors = (float3*)malloc(colorIndexCount*sizeof(float3));

	int idxRow = 0;
	int idxCol = 0;
	float halfHeight = 0.0 ; // (float)0.5/simHeight;
	float halfWidth = 0.0 ; //(float)0.5/simWidth;
	for (int i = 0; i < colorIndexCount; i++){
		if(i%4==0){ //top left of a quad
			quadPoints[i].x = 2.0*(float)idxCol/simWidth - 1.0f + halfWidth;
			quadPoints[i].y = 1.0f - 2.0*(float)idxRow/simHeight - halfHeight;
			quadPoints[i].z = 0.0f;
		}
		if(i%4==1){ //bottom left of a quad
			quadPoints[i].x = 2.0*(float)idxCol/simWidth - 1.0f + halfWidth;
			quadPoints[i].y = 1.0f - 2.0*(float)(idxRow+1)/simHeight - halfWidth;
			quadPoints[i].z = 0.0f;
		}
		if(i%4==2){ //bottom right of a quad
			quadPoints[i].x = 2.0*(float)(idxCol+1)/simWidth - 1.0f + halfWidth;
			quadPoints[i].y = 1.0f - 2.0*(float)(idxRow+1)/simHeight - halfHeight;
			quadPoints[i].z = 0.0f;
		}
		if(i%4==3){ //top right of a quad
			quadPoints[i].x = 2.0*(float)(idxCol+1)/simWidth - 1.0f + halfWidth;
			quadPoints[i].y = 1.0f - 2.0*(float)idxRow/simHeight - halfHeight;
			quadPoints[i].z = 0.0f;
			idxCol++;
		}
		if(idxCol == simWidth){ // row of quads done
			idxCol = 0;
			idxRow++;
		}
	}

	for (int i = 0; i < colorIndexCount; i++){
		colors[i].x = 0.0f;
		colors[i].y = 0.0f; 
		colors[i].z = 0.0f;
	}
			////////*** "Real" Vertices and Velocity data, the size of simDimensions ***/////////

			// I don't bother w/ cuda malloc on host because I don't plan on data transfers

	size_t latticeSize = realIndexCount*sizeof(float2);
	float2 *devPositions; float2 *devVelocities;
	float2 *devVelocities2; 
	float2 *positions = (float2*)malloc(latticeSize);
	float2 *velocities = (float2*)malloc(latticeSize);
	checkCuda(cudaMalloc((void**)&devPositions, latticeSize));
	checkCuda(cudaMalloc((void**)&devVelocities, latticeSize));
	checkCuda(cudaMalloc((void**)&devVelocities2, latticeSize));

	for (int row = 0; row < simHeight; row++){
		for (int col = 0; col < simWidth; col++){
			positions[row*simWidth+col].x = xRange*(float)col/simWidth - xRange/2.0;
			positions[row*simWidth+col].y = yRange/2.0 - yRange*(float)row/simHeight;
			velocities[row*simWidth+col].x = frameVel;
			velocities[row*simWidth+col].y = 0.0;
			if( row == 0 || row == simHeight - 1)
			{
				velocities[row*simWidth+col].x = 0.0;
				velocities[row*simWidth+col].y = 0.0;
			}
			/*
			if( row > 3*simHeight/4 && row < 3*simHeight/4 + 10 && col > simWidth/4 && col < simWidth/4 + 10)
			{
				velocities[row*simWidth+col].x = 5.0;
				velocities[row*simWidth+col].y = 4.0;
			}*/
				
		}
	}
			// TODO special velocity bit for now, visualization */

	/*
	for(int i = 0; i < 5; i++){
		velocities[simWidth * simHeight/2 + i*simWidth + simWidth/2].x = 0.0;
	}*/
	
	float4 boundaries = make_float4(positions[0].x, positions[simWidth*simHeight-1].y,
									positions[simWidth*simHeight-1].x, positions[0].y);
	float dr = xRange/simWidth;
	cout<<"	dR resolution: "<<dr<<" Meters"<<endl;

			/////////*** Load color map data ***/////////

	float3* devColorMap;
	checkCuda(cudaMalloc((void**)&devColorMap, 256*sizeof(float3)));
	float3* colorMap = (float3*)malloc(256*sizeof(float3));
	std::ifstream colorfile("data/Gwyd_Color_Map", ifstream::in);
	std::string line;
	int i = 0;
	while(getline(colorfile, line)){
		std::stringstream linestream(line);
		linestream >> colorMap[i].x >> colorMap[i].y >> colorMap[i].z;
		i++;
	}

			/////////*** Load Obstructed Data ***/////////
	int* obstructed = (int*)malloc(16*sizeof(int));
	int* devObstructed;
	checkCuda(cudaMalloc((void**)&devObstructed, 16*sizeof(int)));

	for(int i = 0; i < 16; i++){
		obstructed[i] = simHeight*(simWidth+i)/2 + simWidth/2 + i;
	}
	testX = simWidth/4 + 1;
	testY = simHeight/2; 
	for(int i = 0; i < 16; i++){
		velocities[obstructed[i]].x = 0.0;
		velocities[obstructed[i]].y = 0.0;
	}

	checkCuda(cudaMemcpy(devObstructed, obstructed, 16*sizeof(int), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(devColorMap, colorMap, 256*sizeof(float3), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(devPositions, positions, latticeSize, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(devVelocities, velocities, latticeSize, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(devVelocities2, velocities, latticeSize, cudaMemcpyHostToDevice));

			/////////*** GLEW Initialization, quarter window ***/////////

	if (!glfwInit()) {
		fprintf(stderr, "ERROR: could not start GLFW3\n");
		return 1;
	} 
	GLFWwindow* window = glfwCreateWindow(1920/(1.5), 1080/(1.5), "Cuda Fluid Dynamics", NULL, NULL);
	glfwSetWindowPos(window, 1920/2, 0);
	if (!window) {
		fprintf(stderr, "ERROR: could not open window with GLFW3\n");
		glfwTerminate();
		return 1;
	}
	glfwMakeContextCurrent(window);
	glewExperimental = GL_TRUE;
	glewInit();

			/////////*** Create position and color vertex buffer objects ***/////////

		// Quad Position Buffer Array

	GLuint pointsVBO = 0;
	glGenBuffers(1, &pointsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, pointsVBO);
	glBufferData(GL_ARRAY_BUFFER, colorIndexCount*sizeof(float3), quadPoints, GL_STATIC_DRAW);

		// Color Buffer Array

	GLuint colorsVBO = 0;
	glGenBuffers(1, &colorsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, colorsVBO);
	glBufferData(GL_ARRAY_BUFFER, colorIndexCount*sizeof(float3), colors, GL_DYNAMIC_DRAW);
	struct cudaGraphicsResource *cudaColorResource;
	checkCuda(cudaGraphicsGLRegisterBuffer(&cudaColorResource, colorsVBO, cudaGraphicsMapFlagsNone));

		// vao binding

	GLuint vertexArray = 0;
	glGenVertexArrays(1, &vertexArray);
	glBindVertexArray(vertexArray);

	glBindBuffer(GL_ARRAY_BUFFER, pointsVBO);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glBindBuffer(GL_ARRAY_BUFFER, colorsVBO);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

			/////////*** Create shader program ***/////////

	GLuint shaderProgram = glCreateProgram();
	glInitShaders(vertexShaderText, fragmentShaderText, shaderProgram);	
	
			/////////*** Draw Loop ***/////////
	
	cudaEvent_t start, stop; 
	float fpsTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	dim3 tpbColor(0, 0);
	dim3 tpbLattice(0, 0);
	dim3 blocks(0, 0);
	initThreadDimensions(simWidth, simHeight, tpbColor, tpbLattice, blocks);
	cout<<"	Calling with Boundaries: "<<boundaries.x<<" "<<boundaries.y<<" "<<boundaries.z<<" "<<boundaries.w<<endl;

	bool thing = true;
	int j = 0;
	while(!glfwWindowShouldClose(window) && thing) {
		cudaEventRecord(start, 0);
		//sleep(1);

			// Run all CUDA kernels including colorization of the linked resource
		if(j==testNum && test == true){
			test = false;
			thing = false;
		}
			
		cout<<endl;
		j++;

		runCuda(&cudaColorResource, devObstructed, devColorMap, 
				devPositions, devVelocities, devVelocities2,
				boundaries, dt, dr, 
				tpbColor, tpbLattice, blocks, 
				simWidth, simHeight, testX, testY, test);
		// TESTING 
		checkCuda(cudaMemcpy(velocities, devVelocities2, latticeSize, cudaMemcpyDeviceToHost));
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glUseProgram(shaderProgram);
		glBindVertexArray(vertexArray);
		glDrawArrays(GL_QUADS, 0, colorIndexCount);

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
	cudaFree(devObstructed);
	cudaFree(devVelocities);
	cudaFree(devPositions);
	cudaFree(devVelocities2);
	cudaFree(devColorMap);
	free(quadPoints);
	free(obstructed);
	free(colorMap);
	free(velocities);
	free(positions);
	free(colors);
	glfwTerminate();
	return 0;
}
