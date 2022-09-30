
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include "cublas_v2.h"

const unsigned int BLOCK_DIM = 32;

cublasHandle_t handle;
cublasStatus_t stat = cublasCreate(&handle);


__global__ void cuda_matmul(const int N , const float *d_A, const float *d_B, float *d_C) 
{
    __shared__ int tile_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ int tile_B[BLOCK_DIM][BLOCK_DIM];

    int row = blockIdx.y * BLOCK_DIM + threadIdx.y;
    int col = blockIdx.x * BLOCK_DIM + threadIdx.x;
    
    if(row >= N || col >= N)
      return;

    float total = 0.0;

    for (int i = 0; i < gridDim.x; i++) 
    {
        int idx = row * N + i * BLOCK_DIM + threadIdx.x;

        tile_A[threadIdx.y][threadIdx.x] = d_A[idx];
      
        idx = (i * BLOCK_DIM + threadIdx.y) * N + col;

        tile_B[threadIdx.y][threadIdx.x] = d_B[idx];
        
        __syncthreads();

        for (int k = 0; k < BLOCK_DIM; ++k)
            total += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
    
        __syncthreads();
    }

    if(row < N && col < N)
        d_C[col * N + row] = total;
}


void matmulCPU(const int N , float *h_A , float *h_B , float *h_C){
  for (int i = 0; i < N; ++i)
    for (int k = 0; k < N; ++k){
      h_C[i + N * k] = 0.f;
      for (int j = 0; j < N; ++j)
        h_C[i + N * k] += h_A[i + j * N] * h_B[j + k * N];
  }
}

void initMat(const int N , float *mat , float val){
  for(unsigned int row = 0 ; row < N ; row++){
    for(unsigned int col = 0 ; col < N ; col++){
      mat[col * N + row] = val;
    }
  }
}

void print_matrix(int N , float *A){
  for(int row = 0 ; row < N ; row++){
    for(int col = 0 ; col < N ; col++){
      printf("%lf " , A[col * N + row]);
    }
    printf("\n");
  }
  printf("\n");
}
// Run the actual benchmark
void benchmark_matmul(const std::size_t N , const unsigned int n_repeat , int mode)
{

  const float val = 2;

  std::size_t totalBytes = N * N * sizeof(float);

  float *h_A = (float*) malloc(totalBytes);
  float *h_B = (float*) malloc(totalBytes);
  float *h_C = (float*) malloc(totalBytes);
  
  initMat(N , h_A , 2);
  initMat(N , h_B , 2);


  float *d_A , *d_B , *d_C;
  cudaMalloc(&d_A, totalBytes);
  cudaMalloc(&d_B, totalBytes);
  cudaMalloc(&d_C, totalBytes);

    
  cudaMemcpy(d_A , h_A , totalBytes ,cudaMemcpyHostToDevice);
  cudaMemcpy(d_B , h_B , totalBytes ,cudaMemcpyHostToDevice);


  const unsigned int n_tests = 3;
  double best = 1e10, worst = 0, avg = 0;
  float alpha = 1.f , beta = 0.;
  
  for (unsigned int t = 0; t < n_tests; ++t){
    // type of t1: std::chrono::steady_clock::time_point
    const auto t1 = std::chrono::steady_clock::now();

    for (unsigned int rep = 0; rep < n_repeat; ++rep){
      if(mode == 0){
        int GRID_DIM = (N + BLOCK_DIM - 1) / BLOCK_DIM;
        dim3 dimGrid(GRID_DIM, GRID_DIM);
        dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
        cuda_matmul<<<dimGrid, dimBlock>>>(N , d_A, d_B, d_C); 
      }
      if(mode == 1){
          stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,&alpha, d_A, N, d_B, N, &beta, d_C, N);
          if (stat != CUBLAS_STATUS_SUCCESS){
            std::cout << "CUBLAS operation failed\n";
            std::abort();
          }
      }
      if(mode == 2){
        matmulCPU(N , h_A , h_B , h_C);
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
  
  if(mode == 0 || mode == 1){
    cudaMemcpy(h_C , d_C, totalBytes, cudaMemcpyDeviceToHost);
  }

  // Copy the result back to the host
  bool wrong_result = false;
  float target_value = val * val * N;
  for(int i = 0 ; i < N * N ; i++)
    if (h_C[i] != target_value)
      wrong_result = true;

  if(wrong_result)
    std::cout << "Error in computation, got something other than "<<target_value<<" for some element\n";

  // Free the memory on the device
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  
  long long ops = 1ll * N * N * N;

  std::cout << "Matrix Matrix Multiplication with "<< N << "rows and " << N <<" columns" 
            << std::setw(8) << 1e-9 * ops / best << " GFlop/s" << std::endl;
}


int main(int argc, char **argv)
{

  if (stat != CUBLAS_STATUS_SUCCESS){
    std::cout << "CUBLAS initialization failed\n";
    std::abort();
  }

  int st = 50 , en = 5000;

  printf("Plain CUDA:: \n");
  for(int n = st ; n <= en ; n = (1 + n * 1.3)){
    n = (n + BLOCK_DIM - 1) / BLOCK_DIM * BLOCK_DIM;
    benchmark_matmul(n , 10 , 0);
  }

  
  printf("CUBLAS :: \n");
  for(int n = st ; n <= en ; n = (1 + n * 1.3)){
    n = (n + BLOCK_DIM - 1) / BLOCK_DIM * BLOCK_DIM;
    benchmark_matmul(n , 10 , 1);
  }

  /*
  printf("Plain CPU:: \n");
  for(int n = st ; n <= min(en , 400) ; n = (1 + n * 1.1)){
    n = (n + BLOCK_DIM - 1) / BLOCK_DIM * BLOCK_DIM;
    benchmark_matmul(n , 5 , 2);
  }*/
  
  cublasDestroy(handle);

  return 0;
}
