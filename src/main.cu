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
	float2 frameVel = make_float2(0.002, 0.0);			// speed of object, right to left


		// testing parameters, TODO to be removed 
	unsigned int testX = simWidth - 10;
	unsigned int testY = simHeight - 10;
	bool test = false;
	int testNum = 300;

	float dt = 10.1;

	int colorIndexCount = simWidth * simHeight * 4;
	int realIndexCount = simWidth * simHeight;
	cout <<"	frameVel will cover "<<frameVel.x*dt/(xRange/simWidth)<<" quads" <<endl;
	int z;
	cin >> z;
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

	size_t latticeFieldSize = realIndexCount*sizeof(float2);
	size_t latticeScalarSize = realIndexCount*sizeof(float);

			// All fields for the device
	// vector fields
	float2 *devPositions; 
	float2 *devVelocities;
	float2 *devVelocities2;
	float2 *devGradPressure;
	// scalar fields
	float *devDivVelocity; 
	float *devPressure;

			// All fields for the Host
	// vector fields
	float2 *positions = (float2*)malloc(latticeFieldSize);
	float2 *velocities = (float2*)malloc(latticeFieldSize);
	float2 *gradPressure = (float2*)malloc(latticeFieldSize);
	// scalar fields
	float *divVelocity = (float*)malloc(latticeScalarSize);
	float *pressure = (float*)malloc(latticeScalarSize);

	// allocate fields for the device
	checkCuda(cudaMalloc((void**)&devPositions, latticeFieldSize));
	checkCuda(cudaMalloc((void**)&devVelocities, latticeFieldSize));
	checkCuda(cudaMalloc((void**)&devVelocities2, latticeFieldSize));
	checkCuda(cudaMalloc((void**)&devGradPressure, latticeFieldSize));


	checkCuda(cudaMalloc((void**)&devDivVelocity, latticeScalarSize));
	checkCuda(cudaMalloc((void**)&devPressure, latticeScalarSize));
	cout << "main 1" << endl;
	for (int row = 0; row < simHeight; row++){
		for (int col = 0; col < simWidth; col++){
			positions[row*simWidth+col].x = xRange*(float)col/simWidth - xRange/2.0;
			positions[row*simWidth+col].y = yRange/2.0 - yRange*(float)row/simHeight;
			velocities[row*simWidth+col].x = frameVel.x;
			velocities[row*simWidth+col].y = frameVel.y;
			divVelocity[row*simWidth+col] = 0.0;
			gradPressure[row*simWidth+col] = make_float2(0.0, 0.0);
			
			/*
			if( row == 0 || row == simHeight - 1)
			{
				velocities[row*simWidth+col].x = 0.0;
				velocities[row*simWidth+col].y = 0.0;
			}
			*/
			pressure[row*simWidth+col] = 0.0;
		}
	}
	cout << "main 2" << endl;


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
		obstructed[i] = (simHeight*(simWidth/2) + simWidth/5 + i*simWidth);
	}
	
	testX = simWidth/4 + 1;
	testY = simHeight/2;
	
	for(int i = 0; i < 16; i++){
		velocities[obstructed[i]].x = 0.0;
		velocities[obstructed[i]].y = 0.0;
	}

	cout << "main 2" << endl;

	checkCuda(cudaMemcpy(devObstructed, obstructed, 16*sizeof(int), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(devColorMap, colorMap, 256*sizeof(float3), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(devPositions, positions, latticeFieldSize, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(devVelocities, velocities, latticeFieldSize, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(devVelocities2, velocities, latticeFieldSize, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(devGradPressure, gradPressure, latticeFieldSize, cudaMemcpyHostToDevice));
	
	checkCuda(cudaMemcpy(devPressure, pressure, latticeScalarSize, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(devDivVelocity, divVelocity, latticeScalarSize, cudaMemcpyHostToDevice));
	cout << "main 2" << endl;


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
	float simTime = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	dim3 tpbColor(0, 0);
	dim3 tpbLattice(0, 0);
	dim3 blocks(0, 0);
	initThreadDimensions(simWidth, simHeight, tpbColor, tpbLattice, blocks);
	cout<<"	Calling with Boundaries: "<<boundaries.x<<" "<<boundaries.y<<" "<<boundaries.z<<" "<<boundaries.w<<endl;

	bool thing = true;
	int j = 0;
	cout << "main 3" << endl;

	while(!glfwWindowShouldClose(window) && thing) {
		cudaEventRecord(start, 0);
		//sleep(1);f

			// Run all CUDA kernels including colorization of the linked resource
		if(j==testNum && test == true){
			test = false;
			thing = false;
		}
			
		cout<<endl;
		j++;

		runCuda(
			&cudaColorResource, 
			devObstructed, 
			devColorMap, 
			devPositions, 
			devVelocities, 
			devVelocities2, 
			devGradPressure, 
			devDivVelocity, 
			devPressure,
			boundaries, 
			dt, 
			dr, 
			frameVel, 
			tpbColor, 
			tpbLattice, 
			blocks, 
			simWidth, 
			simHeight, 
			testX, testY, test);

		// TESTING 
		checkCuda(cudaMemcpy(velocities, devVelocities2, latticeFieldSize, cudaMemcpyDeviceToHost));
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glUseProgram(shaderProgram);
		glBindVertexArray(vertexArray);
		glDrawArrays(GL_QUADS, 0, colorIndexCount);

		glfwSwapBuffers(window);
  		glfwPollEvents();

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&fpsTime, start, stop);
		simTime += dt;
		char title[256];
		sprintf(title, "CudaFluidDynamics: %12.2f fps, simulation time: %12.4f seconds", 1.0f/(fpsTime/1000.0f), simTime);
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
	cudaFree(devDivVelocity);
	cudaFree(devPositions);
	cudaFree(devVelocities2);
	cudaFree(devGradPressure);
	cudaFree(devPressure);
	cudaFree(devColorMap);

	free(divVelocity);
	free(gradPressure);
	free(quadPoints);
	free(obstructed);
	free(colorMap);
	free(velocities);
	free(positions);
	free(colors);

	glfwTerminate();

	return 0;
}
