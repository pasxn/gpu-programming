#include <CL/cl.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <time.h>


cl::Device getDefaultDevice();                            // Return the first device found in this OpenCL platform.
void initializeDevice();                                  // Inicialize device and compile kernel code.
void seqSumArrays(int* a, int* b, int* c, const int N);   // Sequentially performs the N-dimensional operation c = a + b.
void parSumArrays(int* a, int* b, int* c, const int N);   // Parallelly performs the N-dimensional operation c = a + b.
bool checkEquality(int* c1, int* c2, const int N);        // Check if the N-dimensional arrays c1 and c2 are equal.

cl::Program program;    // The program that will run on the device.    
cl::Context context;    // The context which holds the device.    
cl::Device device;      // The device where the kernel will run.

int main() {
    
  // create auxiliary variables
  clock_t start, end;
  const int EXECUTIONS = 10;

  // prepare input arrays
  int ARRAYS_DIM = 1 << 20;
  std::vector<int> a(ARRAYS_DIM, 3);
  std::vector<int> b(ARRAYS_DIM, 5);

  // prepare sequential and parallel outputs
  std::vector<int> cs(ARRAYS_DIM);
  std::vector<int> cp(ARRAYS_DIM);

  // sequentially sum arrays
  start = clock();
  for(int i = 0; i < EXECUTIONS; i++) {
    seqSumArrays(a.data(), b.data(), cs.data(), ARRAYS_DIM);
  }
  end = clock();
  double seqTime = ((double) 10e3 * (end - start)) / CLOCKS_PER_SEC / EXECUTIONS;

  // initialize OpenCL device
  initializeDevice();

  // parallelly sum arrays
  start = clock();
  for(int i = 0; i < EXECUTIONS; i++) {
    parSumArrays(a.data(), b.data(), cp.data(), ARRAYS_DIM);
  }
  end = clock();
  double parTime = ((double) 10e3 * (end - start)) / CLOCKS_PER_SEC / EXECUTIONS;

  // check if outputs are equal
  bool equal = checkEquality(cs.data(), cp.data(), ARRAYS_DIM);

  // print results
  std::cout << "status: " << (equal ? "SUCCESS!" : "FAILED!") << std::endl;
  std::cout << "results: \n\ta[0] = " << a[0] << "\n\tb[0] = " << b[0] << "\n\tc[0] = a[0] + b[0] = " << cp[0] << std::endl;
  std::cout << "mean execution time: \n\tsequential: " << seqTime << " ms;\n\tparallel: " << parTime << " ms." << std::endl;
  std::cout << "performance gain: " << (100 * (seqTime - parTime) / parTime) << "\%\n";
  
  return 0;
}


// return the first device found in this OpenCL platform
cl::Device getDefaultDevice() {
    
  // Search for all the OpenCL platforms available and check if there are any
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  if (platforms.empty()) {
    std::cerr << "no platforms found!" << std::endl;
    exit(1);
  }

  // search for all the devices on the first platform and check if there are any available
  auto platform = platforms.front();
  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

  if (devices.empty()) {
    std::cerr << "no devices found!" << std::endl;
    exit(1);
  }

  // return the first device found
  return devices.front();
}


// inicialize device and compile kernel code
void initializeDevice() {

  // select the first available device
  device = getDefaultDevice();
  
  // read OpenCL kernel file as a string
  std::ifstream kernel_file("add.cl");
  std::string src(std::istreambuf_iterator<char>(kernel_file), (std::istreambuf_iterator<char>()));

  // compile kernel program which will run on the device
  cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));
  context = cl::Context(device);
  program = cl::Program(context, sources);
  
  auto err = program.build();
  if(err != CL_BUILD_SUCCESS) {
    std::cerr << "build status:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) << std::endl;
    std::cerr << "build log   :\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;              
    exit(1);
  }
}


// sequentially performs the N-dimensional operation c = a + b
void seqSumArrays(int* a, int* b, int* c, const int N) {
  for(int i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
  }
}


// parallelly performs the N-dimensional operation c = a + b
void parSumArrays(int* a, int* b, int* c, const int N) {
    
  // create buffers and allocate memory on the device
  cl::Buffer aBuf(context, CL_MEM_READ_ONLY  |  CL_MEM_HOST_NO_ACCESS  |  CL_MEM_COPY_HOST_PTR, N * sizeof(int), a);
  cl::Buffer bBuf(context, CL_MEM_READ_ONLY  |  CL_MEM_HOST_NO_ACCESS  |  CL_MEM_COPY_HOST_PTR, N * sizeof(int), b);
  cl::Buffer cBuf(context, CL_MEM_WRITE_ONLY |  CL_MEM_HOST_READ_ONLY, N * sizeof(int));

  // set kernel arguments
  cl::Kernel kernel(program, "sumArrays");
  kernel.setArg(0, aBuf);
  kernel.setArg(1, bBuf);
  kernel.setArg(2, cBuf);

  // execute the kernel function and collect its result
  cl::CommandQueue queue(context, device);
  queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N));
  queue.enqueueReadBuffer(cBuf, CL_TRUE, 0, N * sizeof(int), c);
}


// check if the N-dimensional arrays c1 and c2 are equal
bool checkEquality(int* c1, int* c2, const int N) {
  for(int i = 0; i < N; i++) {
    if(c1[i] != c2[i]) {
      return false;
    }
  }

  return true;
}
