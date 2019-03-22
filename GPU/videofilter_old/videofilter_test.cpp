#include <stdio.h>
#include <stdlib.h>
#include <iostream> // for standard I/O
#include <fstream>
#include <time.h>
#include "opencv2/opencv.hpp"
#include <math.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#define STRING_BUFFER_LEN 1024

using namespace cv;
using namespace std;
#define PI 3.141592653589
// #define SHOW


void print_clbuild_errors(cl_program program,cl_device_id device){
    cout<<"Program Build failed\n";
    size_t length;
    char buffer[2048];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
    cout<<"--- Build log ---\n "<<buffer<<endl;
    exit(1);
}

unsigned char ** read_file(const char *name) {
    size_t size;
    unsigned char **output=(unsigned char **)malloc(sizeof(unsigned char *));
    FILE* fp = fopen(name, "rb");
    if (!fp) {
        printf("no such file:%s",name);
        exit(-1);
    }

    fseek(fp, 0, SEEK_END);
    size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    *output = (unsigned char *)malloc(size);
    unsigned char **outputstr=(unsigned char **)malloc(sizeof(unsigned char *));
    *outputstr= (unsigned char *)malloc(size);
    if (!*output) {
        fclose(fp);
        printf("mem allocate failure:%s",name);
        exit(-1);
    }

    if(!fread(*output, size, 1, fp)) printf("failed to read file\n");
    fclose(fp);
    printf("file size %d\n",size);
    printf("-------------------------------------------\n");
    snprintf((char *)*outputstr,size,"%s\n",*output);
    printf("%s\n",*outputstr);
    printf("-------------------------------------------\n");
    return outputstr;
}
void callback(const char *buffer, size_t length, size_t final, void *user_data)
{
     fwrite(buffer, 1, length, stdout);
}


void checkError(int status, const char *msg) {
	if(status!=CL_SUCCESS)	
		printf("%s\n",msg);
}

// Randomly generate a floating-point number between -10 and 10.
float rand_float() {
    return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f;
}

void setGaussianKernel(float *kernel, int size){
    double sigma = 0.3*((size-1)*0.5 - 1) + 0.8;
    double sum = 0;
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            kernel[i * size + j] = 1.0 / 2.0 / PI / sigma / sigma * exp(-(i*i+j*j)/2.0/sigma/sigma);
            sum += kernel[i * size + j];
        }
    }
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            kernel[i * size + j] /= sum;
        }
    }
}

void setScharrKernel(unsigned **kernel){

}

int main(int, char**)
{

    VideoCapture camera("./bourne.mp4");
    if(!camera.isOpened())  // check if we succeeded
        return -1;

    const string NAME = "./output.avi";   // Form the new name with container
    int ex = static_cast<int>(CV_FOURCC('M','J','P','G'));
    Size S = Size((int) camera.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
                  (int) camera.get(CV_CAP_PROP_FRAME_HEIGHT));
	//Size S =Size(1280,720);
	cout << "SIZE:" << S << endl;
    
	
    VideoWriter outputVideo;                                        // Open the output
        outputVideo.open(NAME, ex, 25, S, true);

    if (!outputVideo.isOpened())
    {
        cout  << "Could not open the output video for write: " << NAME << endl;
        return -1;
    }
	time_t start,end;
	double diff,tot = 0;
	int count=0;
	// const char *windowName = "filter";   // Name shown in the GUI window.
    // #ifdef SHOW
    // namedWindow(windowName); // Resizable window, might not work on Windows.
    // #endif

    int ROWS = (int) camera.get(CV_CAP_PROP_FRAME_HEIGHT);
    int COLS = (int) camera.get(CV_CAP_PROP_FRAME_WIDTH);
    
    // Declaration of parameters for kernel
    char char_buffer[STRING_BUFFER_LEN];
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_context_properties context_properties[] =
    { 
        CL_CONTEXT_PLATFORM, 0,
        CL_PRINTF_CALLBACK_ARM, (cl_context_properties)callback,
        CL_PRINTF_BUFFERSIZE_ARM, 0x1000,
        0
    };
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    cl_mem input_buf, output_buf, ker_buf;
    cl_event read_event, write_event, kernel_event;
    // cl_ulong start, end;
    int status;
    int errcode;
    printf("hello1\n");

    float *input;
    float *output;
    float *ker_gaussian = NULL;
    int ker_size = 3;
    printf("hello2\n");
    //--------------------------------------------------------------------

    input = (float*)malloc(sizeof(float)*(ROWS)*(COLS));
    output = (float*)malloc(sizeof(float)*ROWS*COLS);
    Scalar value = Scalar(0, 0, 0);
    // Prepare for kernal configurations

    clGetPlatformIDs(1, &platform, NULL);
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
    clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);

    context_properties[1] = (cl_context_properties)platform;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(context_properties, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);

    unsigned char **opencl_program=read_file("videofilter.cl");
    
    program = clCreateProgramWithSource(context, 1, (const char **)opencl_program, NULL, NULL);
    if (program == NULL){
        printf("Program creation failed\n");
        return 1;
    }	
    int success=clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (success!=CL_SUCCESS) print_clbuild_errors(program, device);
    kernel = clCreateKernel(program, "videofilter", NULL);

    // Create buffers 
    input_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
        (ROWS) * (COLS) * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input");
    output_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
        ROWS * COLS * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for output");
    ker_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
        ker_size * ker_size * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for kernel buffer");
    
    // Map kernel buffer
    ker_gaussian = (float *)clEnqueueMapBuffer(queue, ker_buf, CL_FALSE, CL_MAP_WRITE, 0, 
        ker_size * ker_size * sizeof(float), 0, NULL, &write_event, &errcode);    
    checkError(errcode, "Failed to map gaussian kernel");
    
    

    setGaussianKernel(ker_gaussian, ker_size);

    // Unmap buffers 
    errcode = clEnqueueUnmapMemObject(queue, ker_buf, ker_gaussian, 0, NULL, NULL);
    checkError(errcode, "Failed to unmap gaussian kernel");

    
    // Set kernel arguments for execution
    unsigned argi = 0;

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_buf);
    checkError(status, "Failed to set argument 1");

    status = clSetKernelArg(kernel, argi++, sizeof(int), &COLS);
    checkError(status, "Failed to set argument 2");

    status = clSetKernelArg(kernel, argi++, sizeof(int), &ROWS);
    checkError(status, "Failed to set argument 3");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &ker_buf);
    checkError(status, "Failed to set argument 4");

    status = clSetKernelArg(kernel, argi++, sizeof(int), &ker_size);
    checkError(status, "Failed to set argument 5");

    status = clSetKernelArg(kernel, argi++, sizeof(int), &ker_size);
    checkError(status, "Failed to set argument 6");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_buf);
    checkError(status, "Failed to set argument 7");


    
    while (true) {
        Mat cameraFrame,displayframe;
		count = count+1;
		if(count > 299) break;
        camera >> cameraFrame;
        Mat filterframe = Mat(cameraFrame.size(), CV_8UC3);
        Mat grayframe, frame_pad, edge_x, edge_y, edge, edge_inv;
    	cvtColor(cameraFrame, grayframe, CV_BGR2GRAY);
		time (&start);
        
        grayframe.convertTo(grayframe, CV_32FC1);

    	// GaussianBlur(grayframe, grayframe, Size(3,3),0,0);
    	// GaussianBlur(grayframe, grayframe, Size(3,3),0,0);
    	// GaussianBlur(grayframe, grayframe, Size(3,3),0,0);
		// Scharr(grayframe, edge_x, CV_8U, 0, 1, 1, 0, BORDER_DEFAULT );
		// Scharr(grayframe, edge_y, CV_8U, 1, 0, 1, 0, BORDER_DEFAULT );

        
        input = (float *) clEnqueueMapBuffer(queue, input_buf, CL_TRUE, CL_MAP_WRITE, 0, 
            ROWS * COLS* sizeof(float), 0, NULL, &read_event, &errcode);    
        checkError(errcode, "Failed to map input");
        
        memcpy(input, grayframe.data, (ROWS) * (COLS) * sizeof(float));

        errcode = clEnqueueUnmapMemObject(queue, input_buf, input, 0, NULL, NULL);
        checkError(errcode, "Failed to unmap input");


        // Use kernel to realize the convolution
        
        

        // todo tomorrow:
        //      1. add borders to the input matrix
        //      1. write the kernel file: foo(input, width, height, output, edge_arr)
        //      2. get output_buf and edge_buf, and read them back to output and edge_arr arrays
        //      4. convert the output array to grayframe, i.e. edge_arr -> edge

        // Launch the kernel
        const size_t global_work_size[2] = {640, 360};
        const size_t local_work_size[2] = {10, 10};
        status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL,
        global_work_size, local_work_size, 0, NULL, &kernel_event);
        checkError(status, "Failed to launch kernel");
        
        output = (float *) clEnqueueMapBuffer(queue, output_buf, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, 
            ROWS * COLS* sizeof(float), 0, NULL, &read_event, &errcode);    
        checkError(errcode, "Failed to map output");

        // printf("output\n");
        // for (int i=0;i<640;i++){
        //     for (int j=0;j<360;j++){
        //         printf(" %.2f ", output[i*360+j]);
        //     }
        //     printf("\n");
        // }

        





		addWeighted( edge_x, 0.5, edge_y, 0.5, 0, edge );
        threshold(edge, edge, 80, 255, THRESH_BINARY_INV);
		time (&end);

        cvtColor(edge, edge_inv, CV_GRAY2BGR);
    	// Clear the output image to black, so that the cartoon line drawings will be black (ie: not drawn).
    	memset((char*)displayframe.data, 0, displayframe.step * displayframe.rows);
        grayframe = Mat(ROWS, COLS, CV_32FC1, output);

        

        grayframe.convertTo(grayframe, CV_8UC1);
		grayframe.copyTo(displayframe, edge);

        errcode = clEnqueueUnmapMemObject(queue, output_buf, output, 0, NULL, NULL);
        checkError(errcode, "Failed to unmap output");

        cvtColor(displayframe, displayframe, CV_GRAY2BGR);
		outputVideo << displayframe;
	#ifdef SHOW
        imshow(windowName, displayframe);
	#endif
		diff = difftime (end,start);
		tot+=diff;
	}

    
    

	outputVideo.release();
	camera.release();
  	printf ("FPS %.2lf .\n", 299.0/tot );

    

    return EXIT_SUCCESS;

}