// Runcmd : /usr/local/cuda/bin/nvcc cublas_matmul.cu -o matmul -lcublas -lcurand -std=c++14
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <curand.h>
#include <cublas_v2.h>
#include <chrono>

#define MATRIX_M 1024
#define MATRIX_N 512
#define MATRIX_K 256
#define DATATYPE float
#define EPSILON 1e-2

// Is input tranposed => row-majow
// cublas always assumes column major matrices by default
#define A_T false
#define B_T false
#define C_T false

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
      printf("Error at %s:%d\n",__FILE__,__LINE__); \
      exit(-1);}} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
      printf("Error at %s:%d\n",__FILE__,__LINE__);\
      exit(-1);}} while(0)

#define CUBLAS_CALL(x) do { if((x)!=CUBLAS_STATUS_SUCCESS) { \
      printf("Error at %s:%d\n",__FILE__,__LINE__);\
      exit(-1);}} while(0)

// Fill Values using curand
void init_vals(DATATYPE *in, int N)
{
  curandGenerator_t prng;
  CURAND_CALL( curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT) );
  CURAND_CALL( curandSetPseudoRandomGeneratorSeed(prng, 1234ULL) );
  CURAND_CALL( curandGenerateUniform(prng, in, N) );
  CURAND_CALL( curandDestroyGenerator(prng) );
}

// Cublas call
float cublas_matmul(const DATATYPE *A, const DATATYPE *B, DATATYPE *C, const int m, const int n, const int k)
{
  // Events to measure performance
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  int lda = A_T ? k : m;
  int ldb = B_T ? n : k;
  int ldc = C_T ? n : m;
  const DATATYPE alpha = 1;
  const DATATYPE beta = 0;

  // STEP 1: Create cuBLAS Handle
  cublasHandle_t handle;
  CUBLAS_CALL( cublasCreate(&handle) );

  // STEP 2 : Call cuBLAS command
  cudaEventRecord(start);
  if( A_T ) {
    if( B_T ) {
        CUBLAS_CALL( cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc) );
    } else {
        CUBLAS_CALL( cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc) );
    }
  } else{
    if( B_T ) {
        CUBLAS_CALL( cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc) );
    } else {
        CUBLAS_CALL( cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc) );
    }
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);

  // STEP 3 : Destroy Handle
  CUBLAS_CALL( cublasDestroy(handle) );
  return ms;
}

// CPU Verification
inline int get_leading_dimension(int num_rows, int num_cols, bool is_row_major){
    return is_row_major ? num_cols : num_rows;
}

DATATYPE& get_matrix_ref(DATATYPE* matrix, int row, int col, int num_rows, int num_cols, bool is_row_major = false)
{
    int leading_dim = get_leading_dimension(num_rows, num_cols, is_row_major);

    if( is_row_major ) {
        return matrix[ row * leading_dim + col];
    } else {
        return matrix[ col * leading_dim + row];
    }
}

DATATYPE get_matrix_val(const DATATYPE* matrix, int row, int col, int num_rows, int num_cols, bool is_row_major = false)
{
    int leading_dim = get_leading_dimension(num_rows, num_cols, is_row_major);

    if( is_row_major ) {
        return matrix[ row * leading_dim + col];
    } else {
        return matrix[ col * leading_dim + row];
    }
}

DATATYPE cpu_verify(const DATATYPE *A, const DATATYPE *B, DATATYPE *C, const int m_, const int n_, const int k_)
{
  auto cpu_start = std::chrono::steady_clock::now();
  for (int row = 0; row < m_; row++) {
    for (int col = 0; col < n_; col++) {
      DATATYPE& out_c = get_matrix_ref(C, row, col, m_, n_, C_T);
      out_c = 0;
      for (int ki = 0; ki < k_; ki++) {
        DATATYPE in_a = get_matrix_val(A, row, ki, m_, k_, A_T);
        DATATYPE in_b = get_matrix_val(B, ki, col, k_, n_, B_T);
        out_c += in_a * in_b;
      }
    }
  }
  printf("------\n");
  auto cpu_end = std::chrono::steady_clock::now();
  float cpu_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_end - cpu_start).count() * 1e-6;
  return cpu_ms;
}

// Always manages to print in a row-major layout
void print_matrix(const DATATYPE *mat, int num_rows, int num_cols, bool is_row_major = false)
{
  for (int row = 0; row < num_rows; row++) {
    for (int col = 0; col < num_cols; col++) {
        float val = get_matrix_val(mat, row, col, num_rows, num_cols, is_row_major);
        std::cout<< val << " ";
    }
    std::cout << ";\n";
  }
}

int main()
{

    // Declare device side vectors
    thrust::device_vector<DATATYPE> d_A(MATRIX_M * MATRIX_K);
    thrust::device_vector<DATATYPE> d_B(MATRIX_K * MATRIX_N);
    thrust::device_vector<DATATYPE> d_C(MATRIX_M * MATRIX_N);

    // Initialize values using curand
    init_vals(thrust::raw_pointer_cast(d_A.data()), MATRIX_M * MATRIX_K);
    init_vals(thrust::raw_pointer_cast(d_B.data()), MATRIX_K * MATRIX_N);

    // Perform Matrix Multiply on the GPU
    float gpu_time = cublas_matmul(thrust::raw_pointer_cast(d_A.data()),
                    thrust::raw_pointer_cast(d_B.data()),
                    thrust::raw_pointer_cast(d_C.data()),
                    MATRIX_M, MATRIX_N, MATRIX_K);

    // Declare host vectors
    thrust::host_vector<DATATYPE> h_A(MATRIX_M * MATRIX_K);
    thrust::host_vector<DATATYPE> h_B(MATRIX_K * MATRIX_N);
    thrust::host_vector<DATATYPE> h_C(MATRIX_M * MATRIX_N);
    thrust::host_vector<DATATYPE> h_C_computed(MATRIX_M * MATRIX_N);

    // Copy device data to host
    h_A = d_A;
    h_B = d_B;
    h_C_computed = d_C;
      
    // Verify operation on the CPU
    float cpu_time = cpu_verify( thrust::raw_pointer_cast(h_A.data()),
                thrust::raw_pointer_cast(h_B.data()),
                thrust::raw_pointer_cast(h_C.data()),
                MATRIX_M, MATRIX_N, MATRIX_K);

    for(int i = 0; i < MATRIX_M * MATRIX_N; i++){
      if (abs(h_C[i] - h_C_computed[i]) > EPSILON) {
        std::cout << "Mismatch at " << i << " Expected = " << h_C[i] << " Actual = " << h_C_computed[i] << std::endl;

        std::cout << "A :" << std::endl;
        print_matrix( thrust::raw_pointer_cast(h_A.data()), MATRIX_M, MATRIX_K, A_T);
        std::cout << "B :" << std::endl;
        print_matrix( thrust::raw_pointer_cast(h_B.data()), MATRIX_K, MATRIX_N, B_T);
        std::cout << "C Ref :" << std::endl;
        print_matrix( thrust::raw_pointer_cast(h_C.data()), MATRIX_M, MATRIX_N, C_T);
        std::cout << "C Computed :" << std::endl;
        print_matrix( thrust::raw_pointer_cast(h_C_computed.data()), MATRIX_M, MATRIX_N, C_T);
        break;
      }
    }
    std::cout << "TEST COMPLETED \n"
              << "CPU Time : " << cpu_time << " ms\n"
              << "GPU TIme : " << gpu_time << " ms"
              << std::endl;
}
