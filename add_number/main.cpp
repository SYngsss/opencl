#include <iostream>
#include <fstream>
#include <CL/cl.h>
#include <chrono>
#include <cstring>
#define PROGRAM_FILE "add_numbers.cl"
#define KERNEL_FUNC "add_numbers"
using namespace std::chrono;
const int N = 500; // 矩阵大小
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
   // CL_QUEUE_PROFILING_ENABLE 开启计算耗时统计
   cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);
   end = std::chrono::high_resolution_clock::now();
    tt = end - start;
    std::cout << "初始化OpenCL环境 :       " << tt.count() << " s" << std::endl;
    start = std::chrono::high_resolution_clock::now();

    // displayDeviceDetails(devices[i], CL_DEVICE_TYPE/*deviceinfo枚举类型*/, "CL_DEVICE_TYPE");
    // displayDeviceDetails(devices[i], CL_DEVICE_VENDOR, "CL_DEVICE_VENDOR");//CL_DEVICE_VENDOR 用于获取表示设备供应商的字符串。
    // displayDeviceDetails(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, "CL_DEVICE_MAX_COMPUTE_UNITS");
    // displayDeviceDetails(devices[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
    // displayDeviceDetails(devices[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, "CL_DEVICE_MAX_WORK_ITEM_SIZES");
    // displayDeviceDetails(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, "CL_DEVICE_MAX_WORK_GROUP_SIZE");
    // size_t sss;
    // // std::cout <<sizeof(paramsize)/sizeof(float)<< std::endl;
    size_t paramsize1, paramsize, sad;
    cl_int err1 =  CL_SUCCESS;
    err1 = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &sad, nullptr);
    if(err1 != CL_SUCCESS) std::cout <<"get CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS error"<< std::endl;
    else {std::cout <<sad/sizeof(float)<< std::endl;}

	err1 = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &paramsize1, nullptr);
    if(err1 != CL_SUCCESS) std::cout <<"get CL_DEVICE_MAX_COMPUTE_UNITS error"<< std::endl;
    else {std::cout <<paramsize1<< std::endl;}
    
    // err1 = clGetDeviceInfo(device, CL_DEVICE_MAX_SAMPLERS, 0, nullptr, &paramsize);
    // if(err1 != CL_SUCCESS) std::cout <<"get CL_DEVICE_MAX_WORK_ITEM_SIZES error"<< std::endl;
    // else {std::cout <<sizeof(paramsize)/sizeof(float)<< std::endl;}
    // err1 = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, 0, nullptr, &paramsize);
    // if(err1 != CL_SUCCESS) std::cout <<"get CL_DEVICE_MAX_WORK_GROUP_SIZE error"<< std::endl;
    // else {std::cout <<sizeof(paramsize)/sizeof(float)<< std::endl;}
    // char                extensions_buf[4096];
    // std::memset(extensions_buf, 0, sizeof(extensions_buf));
    // clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(extensions_buf), extensions_buf, NULL);
    // std::cout << "Device Extensions: " <<std::string(extensions_buf)<< std::endl;
    
   // 创建OpenCL内存缓冲区
   cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, NULL);
   cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, NULL);
   cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, NULL, NULL);
   
   end = std::chrono::high_resolution_clock::now();
    tt = end - start;
    std::cout << "创建OpenCL内存缓冲区 :       " << tt.count() << " s" << std::endl;
    start = std::chrono::high_resolution_clock::now();
 
   
 
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
    // std::cout << "Program size: " << sizeof(source)/sizeof(const char*) << std::endl;
    cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);
//    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&program_buffer, &program_size, NULL);
   err1 = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
   if(err1 != CL_SUCCESS) std::cout <<"bufferA error"<< std::endl;
   
   cl_kernel kernel = clCreateKernel(program, "add_matrices", NULL);
   free(program_buffer);

   end = std::chrono::high_resolution_clock::now();
    tt = end - start;
    std::cout << "创建OpenCL程序对象 :       " << tt.count() << " s" << std::endl;
    start = std::chrono::high_resolution_clock::now();
 
   // 设置OpenCL内核参数
   err1 = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
   if(err1 != CL_SUCCESS) std::cout <<"bufferA error"<< std::endl;
   err1 = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
   if(err1 != CL_SUCCESS) std::cout <<"bufferA error"<< std::endl;
   err1 = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
   if(err1 != CL_SUCCESS) std::cout <<"bufferA error"<< std::endl;
   
   end = std::chrono::high_resolution_clock::now();
    tt = end - start;
    std::cout << "设置OpenCL内核参数 :       " << tt.count() << " s" << std::endl;
    start = std::chrono::high_resolution_clock::now();

    // 将输入数据传输到OpenCL缓冲区
   err1 = clEnqueueWriteBuffer(queue, bufferA, CL_TRUE, 0, size, A, 0, NULL, NULL);
   if(err1 != CL_SUCCESS) std::cout <<"bufferA error"<< std::endl;
   err1 = clEnqueueWriteBuffer(queue, bufferB, CL_TRUE, 0, size, B, 0, NULL, NULL);
   if(err1 != CL_SUCCESS) std::cout <<"bufferB error"<< std::endl;
 
   // 启动内核
   cl_event event;
   size_t globalWorkSize= N*N*N, localWorkSize = N ;
   err1 = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, &event);
   if(err1 != CL_SUCCESS) std::cout <<"bufferA error"<< std::endl;
   end = std::chrono::high_resolution_clock::now();
    tt = end - start;
    std::cout << "启动内核 :       " << tt.count() << " s" << std::endl;
    start = std::chrono::high_resolution_clock::now();

    // // clWaitForEvents(1, &event);
    // clFinish(queue);
    
    // 在内核执行之后插入事件
    clEnqueueMarkerWithWaitList(queue, 0, NULL, &event);

    // 等待事件完成
    clWaitForEvents(1, &event);

    cl_ulong time_start;
    cl_ulong time_end;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    double nanoSeconds = time_end-time_start;
    // std::cout <<"计算耗时："<< time_start <<"  "<< time_end << std::endl;
    // printf("OpenCl Execution time is: %0.3f milliseconds \n",nanoSeconds / 1000000.0);//转为毫秒级

    // clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, &event);

 
   // 读取结果数据
   err1 = clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, size, A, 0, NULL, NULL);
   if(err1 != CL_SUCCESS) std::cout <<"bufferA error"<< std::endl;
   
   
   end = std::chrono::high_resolution_clock::now();
    tt = end - start;
    std::cout << "读取结果数据 :       " << tt.count() << " s" << std::endl;
    start = std::chrono::high_resolution_clock::now();


    // 将输入数据传输到OpenCL缓冲区
   err1 = clEnqueueWriteBuffer(queue, bufferA, CL_TRUE, 0, size, A, 0, NULL, NULL);
   if(err1 != CL_SUCCESS) std::cout <<"bufferA error"<< std::endl;
   err1 = clEnqueueWriteBuffer(queue, bufferB, CL_TRUE, 0, size, B, 0, NULL, NULL);
   if(err1 != CL_SUCCESS) std::cout <<"bufferB error"<< std::endl;

   end = std::chrono::high_resolution_clock::now();
    tt = end - start;
    std::cout << "再次加载 :       " << tt.count() << " s" << std::endl;
    start = std::chrono::high_resolution_clock::now();

    err1 = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, &event);
    if(err1 != CL_SUCCESS) std::cout <<"bufferA error"<< std::endl;
    end = std::chrono::high_resolution_clock::now();
    tt = end - start;
    std::cout << "再次计算 :       " << tt.count() << " s" << std::endl;
    start = std::chrono::high_resolution_clock::now();

    err1 = clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, size, A, 0, NULL, NULL);
   if(err1 != CL_SUCCESS) std::cout <<"bufferA error"<< std::endl;
   end = std::chrono::high_resolution_clock::now();
    tt = end - start;
    std::cout << "再次输出 :       " << tt.count() << " s" << std::endl;
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

    int numa = 0;
   // 打印结果
   for (int i = 0; i < N * N * N; i++) {
    if(A[i] == 7.0f) numa++;
       
   }
   std::cout << "Result: " << numa << std::endl;
   delete[] A;
   delete[] B;
 
   return 0;
}
