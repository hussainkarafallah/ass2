
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>


const int block_size = 512;
const int chunk_size = 1;

__global__ void compute_triad(const int    N,
  const float  a,
  const float *x,
  const float *y,
  float *      z)
{
  const int idx_base = threadIdx.x + blockIdx.x * (blockDim.x * chunk_size);
  for (unsigned int i = 0; i < chunk_size; ++i)
  {
    const int idx = idx_base + i * block_size;
    if (idx < N)
      z[idx] = a * x[idx] + y[idx];
  }
}

void initVec(const int N , float *vec , const float val){
  for(unsigned int i = 0 ; i < N ; i++)
    vec[i] = val;
}

void initMat(const int M , const int N , float *mat){
  for(unsigned int row = 0 ; row < M ; row++){
    for(unsigned int col = 0 ; col < N ; col++){
      mat[col * N + row] = col;
    }
  }
}

// Run the actual benchmark
void benchmark_triad(const std::size_t M , const std::size_t N , const int repeatBound)
{

  float *h_A = (float*) malloc(M * N * sizeof(float));
  float *h_X = (float*) malloc(N * sizeof(float));

  initVec(N , h_X , 1);
  initMat(M , N , h_A);


  float *d_A , *d_X;
  // allocate matrix and vector
  cudaMalloc(&d_A, M * N * sizeof(float));
  cudaMalloc(&d_X, N * sizeof(float));
  
  cudaMemcpy(d_X , h_X , N * sizeof(float) ,cudaMemcpyHostToDevice);
  cudaMemcpy(d_A , h_X , M * N * sizeof(float) ,cudaMemcpyHostToDevice);


  const unsigned int n_blocks = (N + block_size - 1) / block_size;

  std::vector<float> result_host(N);

  const unsigned int n_tests = 30;
  const unsigned int n_repeat = std::max( 1, (int) (repeatBound / N) );
  double best = 1e10, worst = 0, avg = 0;
  for (unsigned int t = 0; t < n_tests; ++t)
    {
      // type of t1: std::chrono::steady_clock::time_point
      const auto t1 = std::chrono::steady_clock::now();

      //for (unsigned int rep = 0; rep < n_repeat; ++rep)
      //  compute_triad<<<n_blocks, block_size>>>(N, 13.f, v1, v2, v3);

      cudaDeviceSynchronize();
      // measure the time by taking the difference between the time point
      // before starting and now
      const double time =
        std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - t1)
          .count();

      best  = std::min(best, time / n_repeat);
      worst = std::max(worst, time / n_repeat);
      avg += time / n_repeat;
    }

  // Copy the result back to the host
  //cudaMemcpy(result_host.data(), v3, N * sizeof(float), cudaMemcpyDeviceToHost);
  
  if ((result_host[0] + result_host[N - 1]) != 526.f)
    std::cout << "Error in computation, got "
              << (result_host[0] + result_host[N - 1]) << " instead of 526"
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
  if (argc != 5){
      std::cout << "Error, expecting 5 arguments, m_min , m_max, n_min , n_max , repeat";
      std::abort();
  }

  long m_min = static_cast<long>(std::stod(argv[1]));
  long m_max = static_cast<long>(std::stod(argv[2]));
  long n_min = static_cast<long>(std::stod(argv[3]));
  long n_max = static_cast<long>(std::stod(argv[4]));
  long repeat = static_cast<long>(std::stod(argv[5]));


  for(long m = m_min ; m <= m_max ; m = (1 + m * 1.1)){
    for (long n = n_min; n <= n_max; n = (1 + n * 1.1)){
        // round up to nearest multiple of 8
        n = (n + 7) / 8 * 8;
        std::cout<<n<<' '<<m<<endl;
        //benchmark_triad(m , n, repeat);
    }
  }

  return 0;
}
