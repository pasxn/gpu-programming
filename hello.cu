#include <stdio.h>

#define NUM_BLOCKS 16

// number of threads per block
#define BLOCK_WIDTH 1

__global__ void hello() {
  printf("hello, I'm a thread in block %d\n", blockIdx.x);
}

int main(int argc, char** argv) {
  hello<NUM_BLOCKS, BLOCK_WIDTH>>();

  // force the printf()s to flush
  cudaDeviceSynchronize();

  printf("done!\n");

  return 0;
}
