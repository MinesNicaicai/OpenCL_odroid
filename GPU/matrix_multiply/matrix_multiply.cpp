#include <stdio.h>
#include <stdlib.h>
#include <iostream> // for standard I/O
#include <math.h>
#include <time.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <chrono>
#define STRING_BUFFER_LEN 1024
using namespace std;



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

int main()
{
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



    //--------------------------------------------------------------------
    const unsigned M=1024, K=1024, N=1024;
    float *input_A;
    float *input_B;
    float *output;
    float **ref_output=(float **) malloc(sizeof(float*)*M);
    for (unsigned i = 0; i < M; i++){
        ref_output[i] = (float *) malloc(sizeof(float)*N);
    }

    cl_mem input_A_buf; // num_devices elements
    cl_mem input_B_buf; // num_devices elements
    cl_mem output_buf; // num_devices elements
    int status;
    int errcode;

    long long int CPU_time; // in nanosecond
    long long int GPU_map_time; // in nanosecond
    long long int GPU_multiply_time; // in nanosecond
        

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

    unsigned char **opencl_program=read_file("matrix_multiply.cl");
    // unsigned char **opencl_program=read_file("matrix_multiply_group.cl");

    program = clCreateProgramWithSource(context, 1, (const char **)opencl_program, NULL, NULL);
    if (program == NULL){
        printf("Program creation failed\n");
        return 1;
    }	
    int success=clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(success!=CL_SUCCESS) print_clbuild_errors(program,device);
    kernel = clCreateKernel(program, "matrix_multiply", NULL);

    // Create buffers
    // Input buffers.
    input_A_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
        M * K * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input A");

    input_B_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
        K * N * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input B");

    // Output buffer.
    output_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
        M * N * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for output");

    cl_event write_event[2], read_event;
    cl_event kernel_event;
    cl_ulong command_start, command_end;

    auto chrono_start = chrono::steady_clock::now();

    // Map buffers
    input_A = (float *)clEnqueueMapBuffer(queue, input_A_buf, CL_TRUE, CL_MAP_WRITE, 0, M * K * sizeof(float), 0, NULL, &write_event[0], &errcode);    
    checkError(errcode, "Failed to map input A");
    input_B = (float *)clEnqueueMapBuffer(queue, input_B_buf, CL_TRUE, CL_MAP_WRITE, 0, K * N * sizeof(float), 0, NULL, &write_event[1], &errcode);    
    checkError(errcode, "Failed to map input B");
    

    auto chrono_end = chrono::steady_clock::now();
    // printf ("GPU took %lld nanoseconds to map data to the buffers.\n", chrono::duration_cast<chrono::nanoseconds>(chrono_end - chrono_start).count() );

    // Add values to the input arrays

    for (unsigned i = 0; i < M; i++){
        for (unsigned j = 0; j < K; j++){
            input_A[i * K + j] = rand_float();
        }
    }

    for (unsigned i = 0; i < K; i++){
        for (unsigned j = 0; j < N; j++){
            input_B[i * N + j] = rand_float();
        }
    }

    printf("Problem size: M=%u, K=%u, N=%u\n", M, N, K);
    // Run multiplication on CPU
    chrono_start = chrono::steady_clock::now();
    for(unsigned i = 0; i < M; i++) {
        for (unsigned j = 0; j < N; j++){
            float res = 0;
            for (unsigned k = 0; k < K; k++){
                res += input_A[i * K + k] * input_B[k * N + j];
            }
            ref_output[i][j] = res;
        }
    }

    chrono_end = chrono::steady_clock::now();

    CPU_time = chrono::duration_cast<chrono::nanoseconds>(chrono_end - chrono_start).count();
    printf ("CPU took %lld nanoseconds to run matrix multiplication operation.\n", CPU_time);



    // Unmap buffers before execution of kernel
    errcode = clEnqueueUnmapMemObject(queue, input_A_buf, input_A, 0, NULL, NULL);
        checkError(errcode, "Failed to unmap input A");
    errcode = clEnqueueUnmapMemObject(queue, input_B_buf, input_B, 0, NULL, NULL);
        checkError(errcode, "Failed to unmap input B");

    clGetEventProfilingInfo(write_event[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &command_start, NULL);
    clGetEventProfilingInfo(write_event[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &command_end, NULL);
    GPU_map_time = command_end - command_start;
    printf ("GPU took %lld nanoseconds to map data to the buffers. (getEventProfiling)\n", GPU_map_time);

    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.

    // size_t max_work_group_size; 
    // clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
    // printf("max work group size: %d\n", max_work_group_size);

    // clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t), &max_work_group_size, NULL);
    // printf("maximum work-group size: %d\n", max_work_group_size);

    // Set kernel arguments.
    unsigned argi = 0;

    status = clSetKernelArg(kernel, argi++, sizeof(unsigned), &M);
    checkError(status, "Failed to set argument 1");

    status = clSetKernelArg(kernel, argi++, sizeof(unsigned), &K);
    checkError(status, "Failed to set argument 2");

    status = clSetKernelArg(kernel, argi++, sizeof(unsigned), &N);
    checkError(status, "Failed to set argument 3");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_A_buf);
    checkError(status, "Failed to set argument 4");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_B_buf);
    checkError(status, "Failed to set argument 5");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_buf);
    checkError(status, "Failed to set argument 6");

    const size_t global_work_size[2] = {M, N};
    const size_t TS = 16;
    const size_t local_work_size[2] = {TS, TS};

    chrono_start = chrono::steady_clock::now();
    status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL,
        global_work_size, local_work_size, 0, NULL, &kernel_event);
    checkError(status, "Failed to launch kernel");

    // Read the result. This the final operation.
    clWaitForEvents(1, &kernel_event);
    chrono_end = chrono::steady_clock::now();
    
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &command_start, NULL);
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &command_end, NULL);
    GPU_multiply_time = command_end - command_start;
    printf ("GPU took %llu nanoseconds to run matrix multiplication operation.\n", GPU_multiply_time);

    output = (float *)clEnqueueMapBuffer(queue, output_buf, CL_TRUE, CL_MAP_READ, 0, M * N * sizeof(float), 0, NULL, &read_event, &errcode);    
    checkError(errcode, "Failed to map output");
    
    // Verify results.
    bool pass = true;

    for(unsigned i = 0; i < M && pass; i++) {
        for (unsigned j = 0; j < N ; j++){
            if(fabsf(output[i * N + j] - ref_output[i][j]) > 1.0e-5f) {
                printf("Failed verification @ index %d,%d\nOutput: %f\nReference: %f\n",
                i, j, output[i * N + j], ref_output[i][j]);
                pass = false;
            }
        }
    }

    printf("GPU's performance: %.0f MFlops/s \n", (double)(M*N*2*K) / (double)GPU_multiply_time * 1000);
    printf("Speedup = %.2f\n", (double)CPU_time / (double)(GPU_map_time + GPU_multiply_time));



    // Release local events.
    clReleaseEvent(write_event[0]);
    clReleaseEvent(write_event[1]);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseMemObject(input_A_buf);
    clReleaseMemObject(input_B_buf);
    clReleaseMemObject(output_buf);
    clReleaseProgram(program);
    clReleaseContext(context);


    //--------------------------------------------------------------------



    clFinish(queue);

    return 0;
}
