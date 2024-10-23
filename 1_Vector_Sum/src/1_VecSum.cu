#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>
#include <ctime>  // для time()

using namespace std;
using namespace std::chrono;

// CUDA kernel для суммирования элементов вектора
__global__ void vectorSum(const int *vec, unsigned long long *result, int N) {
    __shared__ unsigned long long sharedData[256]; // используем 256 потоков
    int threadId = threadIdx.x;
    int globalId = blockIdx.x * blockDim.x + threadId;

    // Записываем элементы в shared memory
    sharedData[threadId] = (globalId < N) ? vec[globalId] : 0;
    __syncthreads();

    // Сложение элементов в shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadId < stride) {
            sharedData[threadId] += sharedData[threadId + stride];
        }
        __syncthreads();
    }

    // Сохраняем результат в глобальную память
    if (threadId == 0) {
        atomicAdd(result, sharedData[0]);
    }
}

// CPU функция для суммирования элементов вектора
long long cpuVectorSum(const vector<int> &vec) {
    long long sum = 0;
    for (int value : vec) {
        sum += value;
    }
    return sum;
}

int main() {
    srand(time(0));  // инициализируем генератор случайных чисел

    vector<int> sizes = {1000, 1400, 2100, 3000, 4300, 6200, 8900, 12700, 18300, 26400, 37900, 54600, 78500, 112900, 162400, 233600, 336000, 483300, 695200, 1000000}; // размеры векторов

    cout << "+------------+-------------+-------------+---------+------------------+\n";
    cout << "| Vector Size|    CPU Time |    GPU Time | Speedup |  Vector Sum       |\n";
    cout << "+------------+-------------+-------------+---------+------------------+\n";

    for (int N : sizes) {
        vector<int> vec(N);
        generate(vec.begin(), vec.end(), []() { return rand() % 100; });

        // CPU суммирование
        auto start_cpu = high_resolution_clock::now();
        long long cpu_sum = cpuVectorSum(vec);
        auto end_cpu = high_resolution_clock::now();
        duration<double> cpu_time = end_cpu - start_cpu;

        // выделение памяти на GPU
        int *d_vec;
        unsigned long long *d_result;
        cudaMalloc(&d_vec, N * sizeof(int));
        cudaMalloc(&d_result, sizeof(unsigned long long));
        cudaMemcpy(d_vec, vec.data(), N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_result, 0, sizeof(unsigned long long));

        // GPU суммирование
        int THREADS = 256;
        int BLOCKS = (N + THREADS - 1) / THREADS;
        auto start_gpu = high_resolution_clock::now();
        vectorSum<<<BLOCKS, THREADS>>>(d_vec, d_result, N);
        cudaDeviceSynchronize();
        auto end_gpu = high_resolution_clock::now();
        duration<double> gpu_time = end_gpu - start_gpu;

        // копирование результата обратно на хост
        unsigned long long gpu_sum;
        cudaMemcpy(&gpu_sum, d_result, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

        // проверка результата
        if (cpu_sum != gpu_sum) {
            cout << "Ошибка: результаты не совпадают!\n";
            return -1;
        }

        // вычисление ускорения
        double speedup = cpu_time.count() / gpu_time.count();

        // вывод результатов
        cout << fixed << setprecision(6);
        cout << "|" << setw(11) << N << " | "
             << setw(11) << cpu_time.count() << " | "
             << setw(11) << gpu_time.count() << " | "
             << setw(7) << speedup << " | "
             << setw(16) << gpu_sum << " |\n";

        // освобождение памяти на GPU
        cudaFree(d_vec);
        cudaFree(d_result);
    }

    cout << "+------------+-------------+-------------+---------+------------------+\n";
    return 0;
}

