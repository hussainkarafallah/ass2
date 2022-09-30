
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

const unsigned int threads_per_block = 128;

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
  printf("%d %lf\n" , row , result);
  if(row == 0){
    for(int j = 0 ; j < N ; j++)
      printf("%lf " , X[j]);
    printf("\n");
  }
  Y[row] = result;
}

void initVec(const int N , float *vec , const float val){
  for(unsigned int i = 0 ; i < N ; i++)
    vec[i] = val;
}

void initMat(const int M , const int N , float *mat){
  for(unsigned int row = 0 ; row < M ; row++){
    for(unsigned int col = 0 ; col < N ; col++){
      mat[col * M + row] = col;
    }
  }
}

// Run the actual benchmark
void benchmark_triad(const std::size_t M , const std::size_t N , const int repeatBound)
{

  float *h_A = (float*) malloc(M * N * sizeof(float));
  float *h_X = (float*) malloc(N * sizeof(float));
  float *h_Y = (float*) malloc(M * sizeof(float));
  
  initVec(N , h_X , 1);
  initVec(M , h_Y , 0);
  initMat(M , N , h_A);


  float *d_A , *d_X , *d_Y;
  // allocate matrix and vector
  
  cudaMalloc(&d_X, N * sizeof(float));
  cudaMalloc(&d_Y, M * sizeof(float));
  cudaMalloc(&d_A, M * N * sizeof(float));

  for(int row = 0 ; row < M ; row++){
    for(int col = 0 ; col < N ; col++)
      std::cout<<h_A[col * M + row]<<' ';
    std::cout<<'\n';
  }
  
  
  cudaMemcpy(d_X , h_X , N * sizeof(float) ,cudaMemcpyHostToDevice);
  //cudaMemcpy(d_Y , h_Y , M * sizeof(float) ,cudaMemcpyHostToDevice);
  cudaMemcpy(d_A , h_X , M * N * sizeof(float) ,cudaMemcpyHostToDevice);

  
  const unsigned int n_blocks = (M + threads_per_block - 1) / threads_per_block;

  std::vector<float> result_host(M);

  const unsigned int n_tests = 1;
  double best = 1e10, worst = 0, avg = 0;
  
  //const unsigned int n_repeat = std::max( (unsigned int) (1) , (unsigned int) (10000 / M));
  const unsigned int n_repeat = 1;

  for (unsigned int t = 0; t < n_tests; ++t)
    {
      // type of t1: std::chrono::steady_clock::time_point
      const auto t1 = std::chrono::steady_clock::now();

      for (unsigned int rep = 0; rep < n_repeat; ++rep)
        dot_product<<<n_blocks, threads_per_block>>>(M , N , d_A , d_X ,d_Y);

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
  
  float targetResult = N * (N - 1.0) / 2.0;
  if (result_host[0] != targetResult)
    std::cout << "Error in computation, got "
              << result_host[0] << " instead of "<< targetResult
              << std::endl;

  // Free the memory on the device
  cudaFree(d_A);
  cudaFree(d_X);

  std::cout << "STREAM triad of size " << std::setw(8) << N
            << " : min/avg/max: " << std::setw(11) << best << " "
            << std::setw(11) << avg / n_tests << " " << std::setw(11) << worst
            << " seconds or " << std::setw(8) << 1e-6 * N / best
            << " MUPD/s or " << std::setw(8)
            << 1e-9 * 3 * sizeof(float) * N / best << " GB/s" << std::endl;
}

int main(int argc, char **argv)
{
  /*
  arguments:
  m_min
  m_max
  n_min
  n_max
  repeat
  */
  if (argc != 6){
      std::cout << "Error, expecting 5 arguments, m_min , m_max, n_min , n_max , repeat";
      std::abort();
  }

  long m_min = static_cast<long>(std::stod(argv[1]));
  long m_max = static_cast<long>(std::stod(argv[2]));
  long n_min = static_cast<long>(std::stod(argv[3]));
  long n_max = static_cast<long>(std::stod(argv[4]));
  long repeat = static_cast<long>(std::stod(argv[5]));


  for(long m = m_min ; m <= m_max ; m = (1 + m * 1.1)){
    m = (m + 7) / 8 * 8;
    for (long n = n_min; n <= n_max; n = (1 + n * 1.1)){
        // round up to nearest multiple of 8
        n = (n + 7) / 8 * 8;
        std::cout<<n<<' '<<m<<'\n';
        benchmark_triad(m , n, repeat);
    }
  }

  return 0;
}
