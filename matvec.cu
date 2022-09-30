
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include "cublas_v2.h"

const unsigned int threads_per_block = 128;

cublasHandle_t handle;
cublasStatus_t stat = cublasCreate(&handle);


__global__ void dot_product(
  const int M,
  const int N,
  const float *A,
  const float *X,
  float *Y
)
{
  const int row = threadIdx.x + blockIdx.x * blockDim.x;
  if(row >= M)
    return;
  float result = 0.0;
  for(unsigned int col = 0 ; col < N ; col++){
    result += A[col * M + row] * X[col];
  }

  Y[row] = result;
}

void initVec(const int N , float *vec , const float val){
  for(unsigned int i = 0 ; i < N ; i++)
    vec[i] = val;
}

void initMat(const int M , const int N , float *mat , float val){
  for(unsigned int row = 0 ; row < M ; row++){
    for(unsigned int col = 0 ; col < N ; col++){
      mat[col * M + row] = val;
    }
  }
}

// Run the actual benchmark
void benchmark_triad(const std::size_t M , const std::size_t N , const unsigned int n_repeat , int useCublas)
{

  const float val = 97;

  float *h_A = (float*) malloc(M * N * sizeof(float));
  float *h_X = (float*) malloc(N * sizeof(float));
  float *h_Y = (float*) malloc(M * sizeof(float));
  
  initVec(N , h_X , 1);
  initVec(M , h_Y , 0);
  initMat(M , N , h_A , val);


  float *d_A , *d_X , *d_Y;
  // allocate matrix and vector
  
  cudaMalloc(&d_X, N * sizeof(float));
  cudaMalloc(&d_Y, M * sizeof(float));
  cudaMalloc(&d_A, M * N * sizeof(float));
    
  cudaMemcpy(d_X , h_X , N * sizeof(float) ,cudaMemcpyHostToDevice);
  cudaMemcpy(d_A , h_A , M * N * sizeof(float) ,cudaMemcpyHostToDevice);

  
  const unsigned int n_blocks = (M + threads_per_block - 1) / threads_per_block;

  std::vector<float> result_host(M);

  const unsigned int n_tests = 30;
  double best = 1e10, worst = 0, avg = 0;
  

  float alpha = 1.f , beta = 0.;

  for (unsigned int t = 0; t < n_tests; ++t)
    {
      // type of t1: std::chrono::steady_clock::time_point
      const auto t1 = std::chrono::steady_clock::now();

      for (unsigned int rep = 0; rep < n_repeat; ++rep){
        if(useCublas){
          stat =cublasSgemv(handle, CUBLAS_OP_N, M, N, &alpha, d_A, M, d_X, 1, &beta, d_Y, 1);
            if (stat != CUBLAS_STATUS_SUCCESS){
              std::cout << "CUBLAS operation failed\n";
              std::abort();
            }
        }
        else{
          dot_product<<<n_blocks, threads_per_block>>>(M , N , d_A , d_X ,d_Y);
        }
      }

      cudaDeviceSynchronize();
      // measure the time by taking the difference between the time point
      // before starting and now
      const double time =
        std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - t1)
          .count();

      best  = std::min(best, (double) (time / n_repeat));
      worst = std::max(worst, (double) (time / n_repeat));
      avg += time / n_repeat;
    }

  // Copy the result back to the host
  cudaMemcpy(result_host.data(), d_Y, M * sizeof(float), cudaMemcpyDeviceToHost);
  
  float targetResult = N * val;
  if (result_host[0] != targetResult)
    std::cout << "Error in computation, got "
              << result_host[0] << " instead of "<< targetResult
              << std::endl;

  // Free the memory on the device
  cudaFree(d_A);
  cudaFree(d_X);
  
  long long ops = 1ll * N * M;

  std::cout << "matrid vector multiplication with "<< M << " rows and " << N <<" columns" 
            << std::setw(8) << 1e-9 * sizeof(float) * ops / best << " GB/s" << std::endl;
}


int main(int argc, char **argv)
{

  if (stat != CUBLAS_STATUS_SUCCESS){
    std::cout << "CUBLAS initialization failed\n";
    std::abort();
  }

  if (argc != 2){
    std::cout << "Error, add a single argument depending on experiment";
    std::abort();
  }

  long task = static_cast<long>(std::stod(argv[1]));
  // first task measure for n = m for simple parallelization and cublas
  if(task == 1){
    printf("Plain CUDA:: \n");
    for(int n = 100 ; n <= 10000 ; n = (1 + n * 1.1)){
      n = (n + 7) / 8 * 8;
      benchmark_triad(n , n , 10000 / n, 0);
    }
    printf("CUBLAS :: \n");
    for(int n = 100 ; n <= 10000 ; n = (1 + n * 1.1)){
      n = (n + 7) / 8 * 8;
      benchmark_triad(n , n , 10000 / n , 1);
    }
  }
  // measure for constant n for parallelization and cublas
  if(task == 2){

    printf("Plain CUDA:: \n");

    for(int m = 100 ; m <= 10000 ; m = (1 + m * 1.1)){
      m = (m + 7) / 8 * 8;
      benchmark_triad(m , 10000 , 10000 / m , 1);
    }

    printf("CUBLAS:: \n");

    for(int m = 100 ; m <= 10000 ; m = (1 + m * 1.1)){
      m = (m + 7) / 8 * 8;
      benchmark_triad(m , 10000 , 10000 / m , 1);
    }

  }
  // measure for constant m for parallelization and cublas
  if(task == 3){

    printf("Plain CUDA:: \n");

    for(int n = 100 ; n <= 10000 ; n = (1 + n * 1.1)){
      n = (n + 7) / 8 * 8;
      benchmark_triad(16384 , n , 10000 / n , 1);
    }

    printf("CUBLAS:: \n");
    
    for(int n = 100 ; n <= 10000 ; n = (1 + n * 1.1)){
      n = (n + 7) / 8 * 8;
      benchmark_triad(16384 , n , 10000 / n , 1);
    }

  }
  
  
  cublasDestroy(handle);
  
  return 0;
}
