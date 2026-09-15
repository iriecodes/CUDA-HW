// Name: Irie Slabach
// Not simple Julia Set on the GPU
// nvcc G_JuliaExtended.cu -o temp -lglut -lGL

/*
 What to do:
 This code displays a simple Julia set fractal using the GPU.
 However, it currently only runs on a 1024x1024 window.

 Your tasks:
 - Modify the code so it works on any given window size. 
   I will pick these on the fly unsigned int WindowWidth, WindowHeight; 
   float XMin, XMax, YMin, YMax; and your code should work. You will be graded on this.
   
 - But you can set these values to whatever you want for the art compitition.
 - Add color to the fractal — be creative! You will be judged on your artistic flair.
 - Don't cut off your ear or anything, but try to make Vincent wish he'd had a GPU.
 - This is a competition with a prize!!!
*/

/*
 Purpose:
 To have some fun with your new GPU skills!
*/

/*
 Explain what you did to fix the code:

 __global__ void colorPixels(float, float, float, float, float, unsigned int, unsigned int, float); //passing in windowheight and width, art float for the offset

int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

if(pixelX >= windowWidth || pixelY >= windowHeight) //checks within bounds
{
	return;
}

id = 3*(pixelY*windowWidth + pixelX);//works for the rgb

x = xMin + dx * pixelX; // maps coords to the plane
y = yMin + dy * pixelY;

//Threads in a block
//if(WindowWidth > 1024)
//{
// 	printf("The window width is too large to run with this program\n");
// 	printf("The window width must be less than 1024.\n");
// 	printf("Good Bye and have a nice day!\n");
// 	exit(0);
//}

//provides the 256 threads spread across 2d
//blockSize.x = 1024; //WindowWidth;
blockSize.x = 16; //WindowWidth;
//blockSize.y = 1;
blockSize.y = 16;
blockSize.z = 1;

//Blocks in a grid
//gridSize.x = WindowHeight;
gridSize.x = (WindowWidth + blockSize.x - 1) / blockSize.x; //ceiling formula from class for the x direction
//gridSize.y = 1;
gridSize.y = (WindowHeight + blockSize.y - 1) / blockSize.y;; //ceiling formula from class for the y direction
gridSize.z = 1;

TODO: still need to add aspect ratio for fun
const float BaseXMin = XMin;
const float BaseXMax = XMax;
const float BaseYMin = YMin;
const float BaseYMax = YMax;

void reshape(int, int); // added to handle the reshaping of window size and maintain aspect ratio
glutReshapeFunc(reshape);

ART - 
//added math header
#include <math.h>
__device__ void hsvToRgb(float, float, float, float&, float&, float&); for continous color values
//prevents to much leakage added in display()
free(pixelsCPU);
cudaFree(pixelsGPU);

//keeps time for recoloring
void timer(int value)
{
timeOffset += 0.005f;
if (timeOffset > 1.0f) timeOffset -= 1.0f;

glutPostRedisplay();
glutTimerFunc(32, timer, 0); //some amount of fps
}
*/

// Include files
#include <stdio.h>
#include <GL/glut.h>
//added math header
#include <math.h>

// Defines
#define MAXMAG 10.0 // If you grow larger than this, we assume that you have escaped.
#define MAXITERATIONS 200 // If you have not escaped after this many attempts, we assume you are not going to escape.
//#define A  -0.824	//Real part of C
//#define B  -0.1711	//Imaginary part of C
#define A  -0.797	//Real part of C
#define B  -.071	//Imaginary part of C

// Global variables
unsigned int WindowWidth = 1280;
unsigned int WindowHeight = 960;

float XMin = -1.5;
float XMax =  1.5;
float YMin = -1.5;
float YMax =  1.5;

//art for aspect ratio keeps og view as reference
const float BaseXMin = XMin;
const float BaseXMax = XMax;
const float BaseYMin = YMin;
const float BaseYMax = YMax;

//art to allow for color cycling
float timeOffset = 0.0f;

// Function prototypes
void cudaErrorCheck(const char*, int);
__device__ float escapeOrNotColor (float, float);
__device__ void hsvToRgb(float, float, float, float&, float&, float&);
//__global__ void colorPixels(float, float, float, float, float);
__global__ void colorPixels(float, float, float, float, float, unsigned int, unsigned int, float); //passing in windowheight and width, art float for the offset
void display(void);
void reshape(int, int); // added to handle the reshaping of window size and maintain aspect ratio

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
	/*
	if(count < maxCount) 
	{
		return(0.0);
	}
	else
	{
		return(1.0);
	}
	*/
	if(count == maxCount) 
	{
		return -1.0f; // never escaped
	}
	else
	{
		//smoothing
		float smoothCount = (float)count + 1.0f - log(log(mag)) / log(2.0f);
		return smoothCount / (float)maxCount;
	}
}

//continous values for color via hue saturation and value
__device__ void hsvToRgb(float h, float s, float v, float &r, float &g, float &b)
{
	//chroma or color intensity
	float c = v * s;
	//color wheel into sections
	float hh = h * 6.0f;
	//middle values for between colors
	float x = c * (1.0f - fabsf(fmodf(hh, 2.0f) - 1.0f));
	//brightness
	float m = v - c;
	//temp vals waiting for brightness
	float rr, gg, bb;

	if(hh < 1.0f) { rr = c; gg = x; bb = 0.0f; }//red to yellow
	else if 
		(hh < 2.0f) { rr = x; gg = c; bb = 0.0f; }//yellow to green
	else if 
		(hh < 3.0f) { rr = 0.0f; gg = c; bb = x; }//green to cyan
	else if 
		(hh < 4.0f) { rr = 0.0f; gg = x; bb = c; }//cyan to blue
	else if 
		(hh < 5.0f) { rr = x; gg = 0.0f; bb = c; }//blue to magenta
	else                 
		{ rr = c; gg = 0.0f; bb = x; }//magenta to red

	r = rr + m;
	g = gg + m;
	b = bb + m;
}

//__global__ void colorPixels(float *pixels, float xMin, float yMin, float dx, float dy)
__global__ void colorPixels(float *pixels, float xMin, float yMin, float dx, float dy, unsigned int windowWidth, unsigned int windowHeight, float timeOffset)// uses windowWidth and windowHeight now for calculations within the kernel, art timeOffset
{
	float x,y;
	int id;
	
	//Getting the offset into the pixel buffer. 
	//We need the 3 because each pixel has a red, green, and blue value.
	//id = 3*(threadIdx.x + blockDim.x*blockIdx.x);
	
	//Asigning each thread its x and y value of its pixel.
	//x = xMin + dx*threadIdx.x;
	//y = yMin + dy*blockIdx.x;

	int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
	int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

	if(pixelX >= windowWidth || pixelY >= windowHeight) //checks within bounds
	{
		return;
	}

	id = 3*(pixelY*windowWidth + pixelX);//works for the rgb

	x = xMin + dx * pixelX; // maps coords to the plane
	y = yMin + dy * pixelY;

	//art time
	float t = escapeOrNotColor(x, y);

	float r, g, b;
	if (t < 0.0f)
	{
		//inside
		hsvToRgb(fmodf(0.8f - timeOffset * 0.3f, 1.0f), 0.6f, 0.1f, r, g, b);
	}
	else
	{
		//escaped values for smooth and timeoffset
		float hue = fmodf(t * 3.0f + timeOffset, 1.0f);
		hsvToRgb(hue, hue, hue, r, g, b);
	}
	
	//pixels[id] = escapeOrNotColor (x, y);
	//pixels[id+1] = 0.0; //Setting the green
	//pixels[id+2] = 0.0; //Setting the blue 
	pixels[id]   = r;
	pixels[id+1] = g;
	pixels[id+2] = b;
}

void display(void) 
{ 
	dim3 blockSize, gridSize;
	float *pixelsCPU, *pixelsGPU; 
	float stepSizeX, stepSizeY;
	
	//We need the 3 because each pixel has a red, green, and blue value.
	pixelsCPU = (float *)malloc(WindowWidth*WindowHeight*3*sizeof(float));
	cudaMalloc(&pixelsGPU,WindowWidth*WindowHeight*3*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	stepSizeX = (XMax - XMin)/((float)WindowWidth);
	stepSizeY = (YMax - YMin)/((float)WindowHeight);
	
	//Threads in a block
	//if(WindowWidth > 1024)
	//{
	// 	printf("The window width is too large to run with this program\n");
	// 	printf("The window width must be less than 1024.\n");
	// 	printf("Good Bye and have a nice day!\n");
	// 	exit(0);
	//}
	//blockSize.x = 1024; //WindowWidth;
	blockSize.x = 16; //WindowWidth;
	//blockSize.y = 1;
	blockSize.y = 16;
	blockSize.z = 1;
	
	//Blocks in a grid
	//gridSize.x = WindowHeight;
	gridSize.x = (WindowWidth + blockSize.x - 1) / blockSize.x; //ceiling formula from class for the x direction
	//gridSize.y = 1;
	gridSize.y = (WindowHeight + blockSize.y - 1) / blockSize.y;; //ceiling formula from class for the y direction
	gridSize.z = 1;
	
	//colorPixels<<<gridSize, blockSize>>>(pixelsGPU, XMin, YMin, stepSizeX, stepSizeY);
	colorPixels<<<gridSize, blockSize>>>(pixelsGPU, XMin, YMin, stepSizeX, stepSizeY, WindowWidth, WindowHeight, timeOffset);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Copying the pixels that we just colored back to the CPU.
	cudaMemcpyAsync(pixelsCPU, pixelsGPU, WindowWidth*WindowHeight*3*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Putting pixels on the screen.
	glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, pixelsCPU); 
	glFlush(); 

	//prevents to much leakage
	free(pixelsCPU);
	cudaFree(pixelsGPU);
}

//keeps time for recoloring
void timer(int value)
{
	timeOffset += 0.005f;
	if (timeOffset > 1.0f) timeOffset -= 1.0f;

	glutPostRedisplay();
	glutTimerFunc(32, timer, 0); //some amount of fps
}

//maintains aspect ratio based on const variables
void reshape(int w, int h)
{
	//handles minimize
	if (w <= 0) w = 1;
	if (h <= 0) h = 1;

	WindowWidth  = (unsigned int)w;
	WindowHeight = (unsigned int)h;

	//covers window
	glViewport(0, 0, w, h);

	//calculate center points x,y
	float xCenter = 0.5f * (BaseXMin + BaseXMax);
	float yCenter = 0.5f * (BaseYMin + BaseYMax);
	//keeps vertical view
	float halfHeight = 0.5f * (BaseYMax - BaseYMin);
	//prevents stretch
	float halfWidth  = halfHeight * ((float)w / (float)h);

	//calculates new "global" vals but within the function
	XMin = xCenter - halfWidth;
	XMax = xCenter + halfWidth;
	YMin = yCenter - halfHeight;
	YMax = yCenter + halfHeight;

	glutPostRedisplay();
}

int main(int argc, char** argv)
{ 
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WindowWidth, WindowHeight);
	glutCreateWindow("Fractals--Man--Fractals");
   	glutDisplayFunc(display);
	glutReshapeFunc(reshape); // handles resizing
	glutTimerFunc(0, timer, 0);// begins timer
   	glutMainLoop();
}
