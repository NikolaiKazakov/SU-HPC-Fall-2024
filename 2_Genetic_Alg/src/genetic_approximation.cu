#include <iostream>
#include <fstream>
#include <curand_kernel.h>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cuda_runtime.h>

#define POLYNOMIAL_ORDER 4

// Структура для представления индивида (коэффициенты полинома)
struct Individual {
    float coefficients[POLYNOMIAL_ORDER + 1];
};

// Функция для вычисления ошибки аппроксимации
__device__ float fitness(const Individual individual, const float* points_x, const float* points_y, int num_points) {
    float error = 0.0f;
    for (int i = 0; i < num_points; i++) {
        float f_approx = 0.0f;
        for (int j = 0; j <= POLYNOMIAL_ORDER; j++) {
            f_approx += individual.coefficients[j] * powf(points_x[i], j);
        }
        error += powf(f_approx - points_y[i], 2);
    }
    return error;
}

// Ядро для инициализации генератора случайных чисел
__global__ void init_random_state(curandState* devStates, int population_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < population_size) {
        curand_init(clock64(), idx, 0, &devStates[idx]);
    }
}

// Ядро для случайной генерации коэффициентов для индивида
__global__ void generate_population(Individual* population, curandState* devStates, int population_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < population_size) {
        for (int i = 0; i <= POLYNOMIAL_ORDER; i++) {
            population[idx].coefficients[i] = curand_uniform(&devStates[idx]);
        }
    }
}

// Ядро для выполнения мутации
__global__ void mutate_population(Individual* population, curandState* devStates, int population_size, float mutation_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < population_size) {
        for (int i = 0; i <= POLYNOMIAL_ORDER; i++) {
            if (curand_uniform(&devStates[idx]) < mutation_rate) {
                population[idx].coefficients[i] += curand_normal(&devStates[idx]) * 0.1f;
            }
        }
    }
}

// Ядро для выполнения кроссовера
__global__ void crossover_population(Individual* population, curandState* devStates, int population_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < population_size / 2) {
        int parent1 = curand(&devStates[idx]) % population_size;
        int parent2 = curand(&devStates[idx]) % population_size;

        for (int i = 0; i <= POLYNOMIAL_ORDER; i++) {
            if (curand_uniform(&devStates[idx]) < 0.5f) {
                population[parent1].coefficients[i] = population[parent2].coefficients[i];
            }
        }
    }
}

// Ядро для вычисления ошибки аппроксимации всей популяции
__global__ void evaluate_fitness(Individual* population, float* fitness_values, const float* points_x, const float* points_y, int population_size, int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < population_size) {
        fitness_values[idx] = fitness(population[idx], points_x, points_y, num_points);
    }
}

// Ядро для выбора лучшего индивида
__global__ void select_best_individual(Individual* population, float* fitness_values, int population_size, Individual* best_individual, float* best_fitness) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < population_size) {
        if (fitness_values[idx] < *best_fitness) {
            *best_fitness = fitness_values[idx];
            *best_individual = population[idx];
        }
    }
}

// Основная функция
int main(int argc, char* argv[]) {
    if (argc != 7) {
        std::cerr << "Usage: ./genetic_approximation points.txt population_size mutation_mean mutation_variance max_iter max_const_iter\n";
        return -1;
    }

    // Чтение аргументов командной строки
    std::string points_file = argv[1];
    int population_size = std::stoi(argv[2]);
    float mutation_mean = std::stof(argv[3]);
    float mutation_variance = std::stof(argv[4]);
    int max_iter = std::stoi(argv[5]);
    int max_const_iter = std::stoi(argv[6]);

    // Чтение точек из файла
    std::ifstream file(points_file);
    std::vector<float> points_x;
    std::vector<float> points_y;
    float x, y;
    while (file >> x >> y) {
        points_x.push_back(x);
        points_y.push_back(y);
    }
    int num_points = points_x.size();

    // Перенос данных на GPU
    thrust::device_vector<float> dev_points_x = points_x;
    thrust::device_vector<float> dev_points_y = points_y;

    Individual* dev_population;
    float* dev_fitness_values;
    curandState* devStates;
    Individual* dev_best_individual;
    float* dev_best_fitness;

    // Выделение памяти на GPU
    cudaMalloc(&dev_population, population_size * sizeof(Individual));
    cudaMalloc(&dev_fitness_values, population_size * sizeof(float));
    cudaMalloc(&devStates, population_size * sizeof(curandState));
    cudaMalloc(&dev_best_individual, sizeof(Individual));
    cudaMalloc(&dev_best_fitness, sizeof(float));

    // Инициализация состояния генератора случайных чисел
    init_random_state<<<(population_size + 255) / 256, 256>>>(devStates, population_size);
    cudaDeviceSynchronize();

    // Генерация начальной популяции
    generate_population<<<(population_size + 255) / 256, 256>>>(dev_population, devStates, population_size);
    cudaDeviceSynchronize();

    // Для замера времени на GPU
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Основной цикл генетического алгоритма
    int iter = 0;
    int const_iter = 0;
    float best_fitness_val = FLT_MAX;
    Individual best_individual;

    // Инициализация значений для лучшего индивидуума
    cudaMemcpy(dev_best_fitness, &best_fitness_val, sizeof(float), cudaMemcpyHostToDevice);

    while (iter < max_iter && const_iter < max_const_iter) {
        iter++;

        // Оценка ошибки всей популяции
        evaluate_fitness<<<(population_size + 255) / 256, 256>>>(dev_population, dev_fitness_values, thrust::raw_pointer_cast(dev_points_x.data()), thrust::raw_pointer_cast(dev_points_y.data()), population_size, num_points);
        cudaDeviceSynchronize();

        // Селекция лучшего индивида
        select_best_individual<<<(population_size + 255) / 256, 256>>>(dev_population, dev_fitness_values, population_size, dev_best_individual, dev_best_fitness);
        cudaDeviceSynchronize();

        // Копирование лучшего индивида на хост
        cudaMemcpy(&best_fitness_val, dev_best_fitness, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&best_individual, dev_best_individual, sizeof(Individual), cudaMemcpyDeviceToHost);

        // Если улучшений нет, увеличиваем счетчик
        if (best_fitness_val == FLT_MAX) {
            const_iter++;
        } else {
            const_iter = 0;
        }

        // Кроссовер и мутация
        crossover_population<<<(population_size + 255) / 256, 256>>>(dev_population, devStates, population_size);
        mutate_population<<<(population_size + 255) / 256, 256>>>(dev_population, devStates, population_size, mutation_mean);
        cudaDeviceSynchronize();
    }

    // Завершаем замер времени
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Вывод результатов
    std::cout << "Best fitness: " << best_fitness_val << std::endl;
    std::cout << "Best coefficients: ";
    for (int i = 0; i <= POLYNOMIAL_ORDER; i++) {
        std::cout << best_individual.coefficients[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Number of iterations: " << iter << std::endl;
    std::cout << "Time on GPU: " << elapsedTime << " ms" << std::endl;

    // Освобождение памяти
    cudaFree(dev_population);
    cudaFree(dev_fitness_values);
    cudaFree(devStates);
    cudaFree(dev_best_individual);
    cudaFree(dev_best_fitness);

    return 0;
}
