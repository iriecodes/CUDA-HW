// Name: Irie Slabach
// GPU random walk. 
// nvcc P_GPURandomWalk.cu -o temp

/*
 What to do:
 This code runs a random walk for 10,000 steps on the CPU.

 1. Use cuRAND to run 20 random walks simultaneously on the GPU, each with a different seed.
    Print out all 20 final positions.

 2. Use cudaMallocManaged(&variable, amount_of_memory_needed);
    This allocates unified memory, which is automatically managed between the CPU and GPU.
    You lose some control over placement, but it saves you from having to manually copy data
    to and from the GPU.
*/

/*
 Purpose:
 To learn how to use cuRAND and unified memory.
*/

/*
 Explain what you did to fix the code:

 Added curand kernel that runs for 20 random walks in parallel,
 curand_init uses the same seed but different id for randomness,
 x and y change left and right direction based on curand(&state),
 in main i used cudaMallocManaged to print results and you know we are always syncing the cpu and gpu

 // Include files
 #include <curand_kernel.h> // added for curand

 // Defines
 #define NUMBER_OF_WALKS 20 // total number of walks

 // Function prototypes
 __global__ void randomWalkKernel(int*, int*, int, unsigned long long); // kernel for walk

__global__ void randomWalkKernel(int *positionX, int *positionY, int numberOfSteps, unsigned long long seed)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	if(id >= NUMBER_OF_WALKS) return;
	
	curandState state;
	curand_init(seed, id, 0, &state);
	
	int x = 0;
	int y = 0;
	for(int i = 0; i < numberOfSteps; i++)
	{
		if(curand(&state) < 2147483648u) x -= 1;
		else x += 1;
		
		if(curand(&state) < 2147483648u) y -= 1;
		else y += 1;
	}
	
	positionX[id] = x;
	positionY[id] = y;
}

In main - 
	//GPU rando walks
	int *gpuPositionX, *gpuPositionY;
	
	cudaMallocManaged(&gpuPositionX, NUMBER_OF_WALKS*sizeof(int));
	cudaMallocManaged(&gpuPositionY, NUMBER_OF_WALKS*sizeof(int));
	
	randomWalkKernel<<<1, NUMBER_OF_WALKS>>>(gpuPositionX, gpuPositionY, NumberOfRandomSteps, time(NULL));
	cudaDeviceSynchronize();
	
	for(int i = 0; i < NUMBER_OF_WALKS; i++)
	{
		printf("GPU Walk %2d final position = (%d,%d) \n", i, gpuPositionX[i], gpuPositionY[i]);
	}
	
	cudaFree(gpuPositionX);
	cudaFree(gpuPositionY);

 
*/

/*
 Note:
 The maximum signed int value is 2,147,483,647, so the maximum unsigned int value is 4,294,967,295.

 RAND_MAX is guaranteed to be at least 32,767. When I checked it on my laptop (10/6/2025), it was 2,147,483,647.
 rand() returns a value in [0, RAND_MAX]. It actually generates a list of pseudo-random numbers that depends on the seed.
 This list eventually repeats (this is called its period). The period is usually 2³¹ = 2,147,483,648,
 but it may vary by implementation.

 Because RAND_MAX is odd on this machine and 0 is included, there is no exact middle integer.
 Casting to float as in (float)RAND_MAX / 2.0 divides the range evenly.
 Using integer division (RAND_MAX / 2) would bias results slightly toward the positive side by one value out of 2,147,483,647.

 I know this is splitting hares (sorry, rabbits), but I'm just trying to be as accurate as possible.
 You might do this faster with a clever integer approach, but I’m using floats here for clarity.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <curand_kernel.h> // added for curand

// Defines
#define NUMBER_OF_WALKS 20 // total number of walks

// Globals
int NumberOfRandomSteps = 10000;
float MidPoint = (float)RAND_MAX/2.0f;

// Function prototypes
int getRandomDirection();
int main(int, char**);
__global__ void randomWalkKernel(int*, int*, int, unsigned long long); // kernel for walk

__global__ void randomWalkKernel(int *positionX, int *positionY, int numberOfSteps, unsigned long long seed)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	if(id >= NUMBER_OF_WALKS) return;
	
	curandState state;
	curand_init(seed, id, 0, &state);
	
	int x = 0;
	int y = 0;
	for(int i = 0; i < numberOfSteps; i++)
	{
		if(curand(&state) < 2147483648u) x -= 1;
		else x += 1;
		
		if(curand(&state) < 2147483648u) y -= 1;
		else y += 1;
	}
	
	positionX[id] = x;
	positionY[id] = y;
}

int getRandomDirection()
{	
	int randomNumber = rand();
	
	if(randomNumber < MidPoint) return(-1);
	else return(1);
}

int main(int argc, char** argv)
{
	srand(time(NULL));
	
	printf(" RAND_MAX for this implementation is = %d \n", RAND_MAX);
	
	int positionX = 0;
	int positionY = 0;
	for(int i = 0; i < NumberOfRandomSteps; i++)
	{
		positionX += getRandomDirection();
		positionY += getRandomDirection();
	}
	
	printf("\n Final position = (%d,%d) \n", positionX, positionY);

	//GPU rando walks
	int *gpuPositionX, *gpuPositionY;
	
	cudaMallocManaged(&gpuPositionX, NUMBER_OF_WALKS*sizeof(int));
	cudaMallocManaged(&gpuPositionY, NUMBER_OF_WALKS*sizeof(int));
	
	randomWalkKernel<<<1, NUMBER_OF_WALKS>>>(gpuPositionX, gpuPositionY, NumberOfRandomSteps, time(NULL));
	cudaDeviceSynchronize();
	
	for(int i = 0; i < NUMBER_OF_WALKS; i++)
	{
		printf("GPU Walk %2d final position = (%d,%d) \n", i, gpuPositionX[i], gpuPositionY[i]);
	}
	
	cudaFree(gpuPositionX);
	cudaFree(gpuPositionY);
	return 0;
}