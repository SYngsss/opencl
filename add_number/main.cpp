#include <iostream>
#include <fstream>
#include <CL/cl.h>
#include <chrono>
#define PROGRAM_FILE "add_numbers.cl"
#define KERNEL_FUNC "add_numbers"
using namespace std::chrono;
const int N = 100; // 矩阵大小
const size_t size = N * N * N * sizeof(float);
int main() {
   // 初始化输入矩阵
   float* A = new float[N * N * N];
   float* B = new float[N * N * N];
    for (int i = 0; i < N * N * N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N * N * N; i++) {
        A[i] += B[i];
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> tt = end - start;
    std::cout << "CPU      " << tt.count() << " s" << std::endl;

   start = std::chrono::high_resolution_clock::now();
   // 初始化OpenCL环境
   cl_platform_id platform;
   clGetPlatformIDs(1, &platform, NULL);
   cl_device_id device;
   clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
   cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
   cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, NULL);
   end = std::chrono::high_resolution_clock::now();
    tt = end - start;
    std::cout << "初始化OpenCL环境 :       " << tt.count() << " s" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    
    
   // 创建OpenCL内存缓冲区
   cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, NULL);
   cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, NULL);
   cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, NULL, NULL);
   
   end = std::chrono::high_resolution_clock::now();
    tt = end - start;
    std::cout << "创建OpenCL内存缓冲区 :       " << tt.count() << " s" << std::endl;
    start = std::chrono::high_resolution_clock::now();
 
   // 将输入数据传输到OpenCL缓冲区
   clEnqueueWriteBuffer(queue, bufferA, CL_TRUE, 0, size, A, 0, NULL, NULL);
   clEnqueueWriteBuffer(queue, bufferB, CL_TRUE, 0, size, B, 0, NULL, NULL);
 
   end = std::chrono::high_resolution_clock::now();
    tt = end - start;
    std::cout << "将输入数据传输到OpenCL缓冲区 :       " << tt.count() << " s" << std::endl;
    start = std::chrono::high_resolution_clock::now();
 
   // 创建OpenCL程序对象
   const char* source = "__kernel void add_matrices(__global const float* A, __global const float* B, __global float* C) { int id = get_global_id(0); C[id] = A[id] + B[id];}";
   FILE *program_handle;
   size_t program_size;
   char *program_buffer;

   program_handle = fopen("../add_number/add_matrices.cl", "r");
   if(program_handle == NULL) {
      perror("Couldn't find the program file");
      exit(1);
   }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char*)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);
    cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);
//    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&program_buffer, &program_size, NULL);
   clBuildProgram(program, 1, &device, NULL, NULL, NULL);
   cl_kernel kernel = clCreateKernel(program, "add_matrices", NULL);
   free(program_buffer);
   
   end = std::chrono::high_resolution_clock::now();
    tt = end - start;
    std::cout << "创建OpenCL程序对象 :       " << tt.count() << " s" << std::endl;
    start = std::chrono::high_resolution_clock::now();
 
   // 设置OpenCL内核参数
   clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
   clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
   clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
   
   end = std::chrono::high_resolution_clock::now();
    tt = end - start;
    std::cout << "设置OpenCL内核参数 :       " << tt.count() << " s" << std::endl;
    start = std::chrono::high_resolution_clock::now();
 
   // 启动内核
   size_t globalWorkSize[2] = { N*N*N, 1 };
   clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
   
   end = std::chrono::high_resolution_clock::now();
    tt = end - start;
    std::cout << "启动内核 :       " << tt.count() << " s" << std::endl;
    start = std::chrono::high_resolution_clock::now();
 
   // 读取结果数据
   clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, size, A, 0, NULL, NULL);
   
   end = std::chrono::high_resolution_clock::now();
    tt = end - start;
    std::cout << "读取结果数据 :       " << tt.count() << " s" << std::endl;
    start = std::chrono::high_resolution_clock::now();
 
   // 清理OpenCL资源
   clReleaseMemObject(bufferA);
   clReleaseMemObject(bufferB);
   clReleaseMemObject(bufferC);
   clReleaseProgram(program);
   clReleaseKernel(kernel);
   clReleaseCommandQueue(queue);
   clReleaseContext(context);
   
   end = std::chrono::high_resolution_clock::now();
    tt = end - start;
    std::cout << "清理OpenCL资源 :       " << tt.count() << " s" << std::endl;
 
   // 打印结果
   for (int i = 0; i < N * N * N; i++) {
       std::cout << "Result: " << A[i] << std::endl;
   }
   
   delete[] A;
   delete[] B;
 
   return 0;
}
