
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include "cublas_v2.h"

const unsigned int threads_per_block = 1024;

cublasHandle_t handle;
cublasStatus_t stat = cublasCreate(&handle);

/*
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
}*/

void initVec(const int N , float *vec , const float val){
  for(unsigned int i = 0 ; i < N ; i++)
    vec[i] = val;
}

void initMat(const int M , const int N , float *mat){
  for(unsigned int row = 0 ; row < M ; row++){
    for(unsigned int col = 0 ; col < N ; col++){
      mat[col * M + row] = (row / 100);
    }
  }
}

// Run the actual benchmark
void benchmark_triad(const std::size_t M , const std::size_t N , const unsigned int n_repeat , int useCublas)
{

  const float val = 97;

  float *h_A = (float*) malloc(M * N * sizeof(float));
  float *h_X = (float*) malloc(M * sizeof(float));
  float *h_Y = (float*) malloc(N * sizeof(float));
  
  // initialize vector to all ones
  initVec(M , h_X , 1);
  // initialize matrix so ith row has integer part of (i/100)
  // first 100 columns will be zeroes, second 100 columns will be ones ... etc
  initMat(M , N , h_A);
  // expected result of ith scalar of resulting vector is M * int(i/100)
  initVec(N , h_Y , 0);

  float *d_A , *d_X , *d_Y;

  cudaMalloc(&d_X, M * sizeof(float));
  cudaMalloc(&d_Y, N * sizeof(float));
  cudaMalloc(&d_A, M * N * sizeof(float));
    
  cudaMemcpy(d_X , h_X , M * sizeof(float) ,cudaMemcpyHostToDevice);
  cudaMemcpy(d_A , h_A , M * N * sizeof(float) ,cudaMemcpyHostToDevice);

  
  std::vector<float> result_host(N);

  const unsigned int n_tests = 30;
  double best = 1e10, worst = 0, avg = 0;
  

  float alpha = 1.f , beta = 0.;

  for (unsigned int t = 0; t < n_tests; ++t)
    {
      // type of t1: std::chrono::steady_clock::time_point
      const auto t1 = std::chrono::steady_clock::now();

      for (unsigned int rep = 0; rep < n_repeat; ++rep){
        if(useCublas){
          stat =cublasSgemv(handle, CUBLAS_OP_T, M, N, &alpha, d_A, M, d_X, 1, &beta, d_Y, 1);
            if (stat != CUBLAS_STATUS_SUCCESS){
              std::cout << "CUBLAS operation failed\n";
              std::abort();
            }
        }
        else{
          const unsigned int n_blocks = (M + threads_per_block - 1) / threads_per_block;
          //dot_product<<<n_blocks, threads_per_block>>>(M , N , d_A , d_X ,d_Y);
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
  cudaMemcpy(result_host.data(), d_Y, N * sizeof(float), cudaMemcpyDeviceToHost);
  
  int bad_result = 0;
  for(int i = 0 ; i < N ; i++){
    int expected = (i / 100) * M;
    if(result_host[i] != expected)
      bad_result = 1;
  }

  for(int i = 0 ; i < M ; i++){
    for(int j = 0 ; j < N ; j++){
      printf("%lf " , h_A[j * M + i]);
    }
    printf("\n");
  }
  for(int j = 0 ; j < N ; j++){
    printf("%lf " , h_X[j]);
  }
  printf("\n");

  if (bad_result)
    std::cout << "Error in computation, some scalar in the vector is not as expected" << std::endl;

  // Free the memory on the device
  cudaFree(d_A);
  cudaFree(d_X);
  
  long long ops = 1ll * N * M;

  std::cout << "STREAM triad with "<< M << "rows and " << N <<" columns" 
            << std::setw(8) << 1e-6 * ops / best << " MUPD/s or " 
            << std::setw(8) << 1e-9 * sizeof(float) * ops / best << " GB/s" << std::endl;
}


int main(int argc, char **argv)
{

  if (stat != CUBLAS_STATUS_SUCCESS){
    std::cout << "CUBLAS initialization failed\n";
    std::abort();
  }


  //printf("Plain CUDA:: \n");
  //benchmark_triad(5000 , 5000 , 30, 0);
  
  printf("CUBLAS :: \n");
  for(int n = 32 ; n <= 32 ; n = (1 + n * 1.1)){
     benchmark_triad(n , n , 1 , 1);
  }
  
  return 0;
}
