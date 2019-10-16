#include <stdio.h>

// This function (aka "Kernel") runs on the GPU
__global__ void hello_world()
{
   printf("Hello World from Thread %d !\n", threadIdx.x);
}

int main(void)
{

  // Run hello World on the GPU
  hello_world<<<1, 1>>>();

  // Wait for GPU to finish
  cudaDeviceSynchronize();

  return 0;
}
