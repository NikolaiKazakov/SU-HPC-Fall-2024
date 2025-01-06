#include <iostream>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include "EasyBMP.hpp"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>    

#define M_PI 3.14159265358979323846    // Определение значения числа пи

// Структура для представления трехмерных векторов
struct Vec3 {
    float x, y, z;
    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    __host__ __device__ Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    __host__ __device__ Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    __host__ __device__ Vec3 operator*(float scalar) const { return Vec3(x * scalar, y * scalar, z * scalar); }
    __host__ __device__ Vec3 operator/(float scalar) const { return Vec3(x / scalar, y / scalar, z / scalar); }
    __host__ __device__ float dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    __host__ __device__ Vec3 normalize() const {
        float len = sqrt(x * x + y * y + z * z);    // Нормализация вектора
        return *this / len;
    }
};

// Структура для представления луча
struct Ray {
    Vec3 origin, direction;
    __host__ __device__ Ray(const Vec3& o, const Vec3& d) : origin(o), direction(d) {}
};

// Структура для представления сферы
struct Sphere {
    Vec3 center;
    float radius;
    Vec3 color;
    bool reflective;
    __host__ __device__ Sphere() : center(Vec3()), radius(0), color(Vec3()), reflective(false) {}
    __host__ __device__ Sphere(const Vec3& c, float r, const Vec3& col, bool refl)
        : center(c), radius(r), color(col), reflective(refl) {}
    __host__ __device__ bool intersect(const Ray& ray, float& t) const {
        Vec3 oc = ray.origin - center;
        float b = 2.0f * oc.dot(ray.direction);
        float c = oc.dot(oc) - radius * radius;
        float discriminant = b * b - 4.0f * c;    // Проверка на пересечение луча со сферой
        if (discriminant > 0) {
            t = (-b - sqrt(discriminant)) / 2.0f;
            return t > 0;
        }
        return false;
    }
};

// Трассировка луча для нахождения цвета
__device__ Vec3 traceRay(const Ray& ray, int depth, const Sphere* spheres, int numSpheres, const Vec3* lightSources, int numLights, curandState* state) {
    if (depth > 5) return Vec3(0, 0, 0);    // Ограничение глубины рекурсии
    float t_min = 1e20;
    const Sphere* hitSphere = nullptr;
    float t_hit = 0;
    for (int i = 0; i < numSpheres; i++) {
        float t = 0;
        if (spheres[i].intersect(ray, t) && t < t_min) {
            t_min = t;
            hitSphere = &spheres[i];
            t_hit = t;
        }
    }
    if (hitSphere) {
        Vec3 hitPoint = ray.origin + ray.direction * t_hit;
        Vec3 normal = (hitPoint - hitSphere->center).normalize();
        Vec3 color = hitSphere->color;
        Vec3 illumination = Vec3(0, 0, 0);
        for (int i = 0; i < numLights; i++) {
            Vec3 lightDir = (lightSources[i] - hitPoint).normalize();
            float lightIntensity = fmax(0.0f, normal.dot(lightDir));
            illumination = illumination + color * lightIntensity;    // Добавление освещения от источников
        }
        if (hitSphere->reflective && depth < 5) {
            Vec3 reflectionDir = ray.direction - normal * 2.0f * ray.direction.dot(normal);
            Ray reflectionRay(hitPoint + reflectionDir * 0.001f, reflectionDir);    // Отраженный луч
            illumination = illumination + traceRay(reflectionRay, depth + 1, spheres, numSpheres, lightSources, numLights, state);
        }
        return illumination;
    }
    return Vec3(0, 0, 0);    // Возврат черного цвета при отсутствии пересечения
}

// Основная функция рендеринга изображения
__global__ void renderImage(Vec3* pixels, Sphere* spheres, int numSpheres, Vec3* lightSources, int numLights, curandState* states, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    curandState state = states[y * width + x];
    curand_init(1234, x + y * width, 0, &state);    // Инициализация состояния генератора случайных чисел
    float aspect_ratio = float(width) / float(height);
    float fov = 90.0f;
    float angle = tan(fov * 0.5f * M_PI / 180.0f);
    float px = (2 * (x + 0.5f) / float(width) - 1) * angle * aspect_ratio;
    float py = (1 - 2 * (y + 0.5f) / float(height)) * angle;
    Vec3 rayDir(px, py, -1);
    rayDir = rayDir.normalize();
    Ray ray(Vec3(0, 2, 5.66), rayDir);     // Положение камеры
    pixels[y * width + x] = traceRay(ray, 0, spheres, numSpheres, lightSources, numLights, &state);
    states[y * width + x] = state;
}

// Генерация случайной позиции для источника света или объекта
Vec3 generateRandomPositionOnCircle(float radius, float height = 0.0f) {
    float angle = static_cast<float>(rand()) / RAND_MAX * 2 * M_PI;
    return Vec3(radius * cos(angle), height, radius * sin(angle));
}

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: ./ray_trace <numSpheres> <numLights> <width> <height> <imagename>" << std::endl;
        return 1;
    }

    // Чтение параметров командной строки
    int numSpheres = atoi(argv[1]);
    int numLights = atoi(argv[2]);
    int width = atoi(argv[3]);
    int height = atoi(argv[4]);
    const char* imagename = argv[5];

    srand(time(0));
    Sphere* spheres = new Sphere[numSpheres];
    for (int i = 0; i < numSpheres; i++) {
        float radius = 0.4f + static_cast<float>(rand()) / RAND_MAX * 0.6f;
        Vec3 position = generateRandomPositionOnCircle(4, 0);
        Vec3 color(static_cast<float>(rand()) / RAND_MAX, static_cast<float>(rand()) / RAND_MAX, static_cast<float>(rand()) / RAND_MAX);
        bool reflective = rand() % 2;
        spheres[i] = Sphere(position, radius, color, reflective);
    }

    Vec3* lightSources = new Vec3[numLights];
    for (int i = 0; i < numLights; i++) {
        lightSources[i] = generateRandomPositionOnCircle(5.66f, 1.0f + static_cast<float>(rand()) / RAND_MAX);
    }

    // Выделение памяти на устройстве
    Sphere* d_spheres;
    Vec3* d_lightSources;
    Vec3* d_pixels;
    curandState* d_states;

    cudaMalloc((void**)&d_spheres, sizeof(Sphere) * numSpheres);
    cudaMalloc((void**)&d_lightSources, sizeof(Vec3) * numLights);
    cudaMalloc((void**)&d_pixels, sizeof(Vec3) * width * height);
    cudaMalloc((void**)&d_states, sizeof(curandState) * width * height);

    cudaMemcpy(d_spheres, spheres, sizeof(Sphere) * numSpheres, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lightSources, lightSources, sizeof(Vec3) * numLights, cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    renderImage<<<gridSize, blockSize>>>(d_pixels, d_spheres, numSpheres, d_lightSources, numLights, d_states, width, height);

    Vec3* pixels = new Vec3[width * height];
    cudaMemcpy(pixels, d_pixels, sizeof(Vec3) * width * height, cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    // Сохранение изображения
    EasyBMP::Image image(width, height);
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            EasyBMP::RGBColor color;
            color.SetColor(
                std::min(int(pixels[j * width + i].x * 255), 255),
                std::min(int(pixels[j * width + i].y * 255), 255),
                std::min(int(pixels[j * width + i].z * 255), 255)
            );
            image.SetPixel(i, j, color);
        }
    }

    image.Write(imagename);

    delete[] pixels;
    delete[] spheres;
    delete[] lightSources;
    cudaFree(d_spheres);
    cudaFree(d_lightSources);
    cudaFree(d_pixels);
    cudaFree(d_states);

    return 0;
}
