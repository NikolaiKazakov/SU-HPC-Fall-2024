#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>

using std::cout;
using std::generate;
using std::vector;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

__global__ void matrixMul(const int *a, const int *b, int *c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        c[row * N + col] = 0;
        for (int k = 0; k < N; k++) {
            c[row * N + col] += a[row * N + k] * b[k * N + col];
        }
    }
}

void cpuMatrixMul(const vector<int> &a, const vector<int> &b, vector<int> &c, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            c[i * N + j] = 0;
            for (int k = 0; k < N; k++) {
                c[i * N + j] += a[i * N + k] * b[k * N + j];
            }
        }
    }
}

void verify_result(vector<int> &a, vector<int> &b, vector<int> &c, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int tmp = 0;
            for (int k = 0; k < N; k++) {
                tmp += a[i * N + k] * b[k * N + j];
            }
            assert(tmp == c[i * N + j]);
        }
    }
}

int main() {
    // размеры матриц от 100 до 2000 с шагом 100
    vector<int> matrix_sizes;
    for (int size = 100; size <= 2000; size += 100) {
        matrix_sizes.push_back(size);
    }

    cout << "+-----------+-------------+-------------+----------+\n";
    cout << "|Matrix Size|    CPU      |    GPU      | Speedup  |\n";
    cout << "+-----------+-------------+-------------+----------+\n";

    for (int N : matrix_sizes) {
        size_t bytes = N * N * sizeof(int);

        // векторы для хранения матриц
        vector<int> h_a(N * N), h_b(N * N), h_c(N * N), h_c_cpu(N * N);

        // инициализация матриц случайными значениями
        generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
        generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

        // выделение памяти GPU
        int *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);

        // копирование данных на устройство
        cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

        // устанавливаем размерность блоков и сетки
        int THREADS = 32;
        int BLOCKS = (N + THREADS - 1) / THREADS;  
        dim3 threads(THREADS, THREADS);
        dim3 blocks(BLOCKS, BLOCKS);

        // измерение времени на CPU
        auto start_cpu = high_resolution_clock::now();
        cpuMatrixMul(h_a, h_b, h_c_cpu, N);
        auto end_cpu = high_resolution_clock::now();
        duration<double> cpu_time = end_cpu - start_cpu;

        // измерение времени на GPU
        auto start_gpu = high_resolution_clock::now();
        matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize(); 
        auto end_gpu = high_resolution_clock::now();
        duration<double> gpu_time = end_gpu - start_gpu;

        // копирование результата с устройства на хост
        cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

        // проверка корректности результата
        verify_result(h_a, h_b, h_c, N);

        // вычисление ускорения
        double speedup = cpu_time.count() / gpu_time.count();

        // вывод таблицы
        cout << std::fixed << std::setprecision(6);
        cout << "|" << std::setw(11) << N << "x" << N << " | "
             << std::setw(11) << cpu_time.count() << " | "
             << std::setw(11) << gpu_time.count() << " | "
             << std::setw(8) << speedup << " |\n";
        cout << "+-----------+-------------+-------------+----------+\n";

        // освобождение памяти на устройстве
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

    return 0;
}
