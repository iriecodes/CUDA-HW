// Name: Irie Slabach
// Simple Julia CPU.
// nvcc F_JuliaCPUtoGPU.cu -o temp -lglut -lGL
// glut and GL are openGL libraries.
/*
 What to do:
 This code displays a simple Julia fractal using the CPU.
 Rewrite the code so that it uses the GPU to create the fractal. 
 Keep the window at 1024 by 1024.
 Use __device__ for the escapeOrNotColor function
*/

/*
 Purpose:
 To apply your new GPU skills to do  something cool!
*/

/*
 Explain what you did to fix the code:
 Added -

	dim3 BlockSize;
	dim3 GridSize;
	float *Pixels_GPU;
	float *Pixels_CPU;

	void setUpDevices();
	void allocateMemory();
	void cleanUp();
	//float escapeOrNotColor(float, float);
	__device__ float escapeOrNotColor(float, float); - converted from cpu version to gpu
	__global__ void fractal(float*, float, float, float, float, int, int); - gpu kernel that computes one pixel of the fractal per thread code explained inline
 
*/

// Include files
#include <stdio.h>
#include <GL/glut.h>

// Defines
#define MAXMAG 10.0 // If you grow larger than this, we assume that you have escaped.
#define MAXITERATIONS 200 // If you have not escaped after this many attempts, we assume you are not going to escape.
#define A  -0.824	//Real part of C
#define B  -0.1711	//Imaginary part of C

// Global variables
unsigned int WindowWidth = 1024;
unsigned int WindowHeight = 1024;

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

dim3 BlockSize;
dim3 GridSize;
float *Pixels_GPU;
float *Pixels_CPU;

// Function prototypes
void cudaErrorCheck(const char*, int);
void setUpDevices();
void allocateMemory();
void cleanUp();
//float escapeOrNotColor(float, float);
__device__ float escapeOrNotColor(float, float);
__global__ void fractal(float*, float, float, float, float, int, int);
void display(void);


void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

void setUpDevices()
{
	BlockSize.x = 1024;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = 1024;
	GridSize.y = 1;
	GridSize.z = 1;
}

void allocateMemory()
{
	cudaMalloc(&Pixels_GPU, WindowWidth*WindowHeight*3*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);

	Pixels_CPU = (float *)malloc(WindowWidth*WindowHeight*3*sizeof(float));
}

void cleanUp()
{
	cudaFree(Pixels_GPU);
	free(Pixels_CPU);
}

__device__ float escapeOrNotColor (float x, float y) 
{
	float mag,tempX;
	int count;
	
	int maxCount = MAXITERATIONS;
	float maxMag = MAXMAG;
	
	count = 0;
	mag = sqrt(x*x + y*y);;
	while (mag < maxMag && count < maxCount) 
	{	
		tempX = x; //We will be changing the x but we need its old value to find y.
		x = x*x - y*y + A;
		y = (2.0 * tempX * y) + B;
		mag = sqrt(x*x + y*y);
		count++;
	}
	if(count < maxCount) 
	{
		return(0.0);
	}
	else
	{
		return(1.0);
	}
}

__global__ void fractal(float *pixels, float xMin, float yMin, float stepSizeX, float stepSizeY, int width, int height)
{

	int id = threadIdx.x + blockIdx.x * blockDim.x;
	
	//1d to 2d
	int row = id/width; //row
	int col = id%width; //col
	//pixel mapping
	float x = xMin + col*stepSizeX;
	float y = yMin + row*stepSizeY;

	int k = 3 * (row*width+col);
	
	//does it escape?
	float color = escapeOrNotColor(x,y);
	
	//make it green if it stays
	pixels[k] = color*0; //Red
	pixels[k+1] = color*1.0; //Green
	pixels[k+2] = color*0; //Blue
}

void display(void) 
{ 
	float stepSizeX, stepSizeY;
	
	stepSizeX = (XMax-XMin)/((float)WindowWidth);
	stepSizeY = (YMax-YMin)/((float)WindowHeight);
	
	fractal<<<GridSize, BlockSize>>>(Pixels_GPU, XMin, YMin, stepSizeX, stepSizeY, WindowWidth, WindowHeight);
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaMemcpy(Pixels_CPU, Pixels_GPU, WindowWidth*WindowHeight*3*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, Pixels_CPU); 
	glFlush(); 
}

int main(int argc, char** argv)
{ 

	setUpDevices();
	allocateMemory();
	
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WindowWidth, WindowHeight);
	glutCreateWindow("Fractals--Man--Fractals");
   	glutDisplayFunc(display);
   	glutMainLoop();
   	
   	
   	cleanUp();

   	return(0);
}