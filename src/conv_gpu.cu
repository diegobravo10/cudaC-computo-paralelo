#include "conv_gpu.h"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

void verificarCuda(int codigo, const char* archivo, int linea) {
    if (codigo != cudaSuccess) {
        std::fprintf(stderr, "Error CUDA %d (%s) en %s:%d\n", codigo,
                     cudaGetErrorString((cudaError_t)codigo), archivo, linea);
        std::exit(1);
    }
}

// Cada hilo calcula UN píxel de salida.
// Distribución 1D: el índice global recorre la imagen fila por fila,
//   idx  = blockIdx.x * blockDim.x + threadIdx.x
//   fila = idx / ancho,  col = idx % ancho
// Los hilos con idx >= ancho*alto (sobrantes del último bloque) no hacen nada.
__global__ void convolucionKernel(const float* __restrict__ entrada,
                                  float* __restrict__ salida,
                                  const float* __restrict__ mascara,
                                  int ancho, int alto, int k) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ancho * alto) return;  // control de límites

    const int fila = idx / ancho;
    const int col = idx % ancho;
    const int r = k / 2;

    float suma = 0.0f;
    for (int i = 0; i < k; ++i) {
        const int y = fila + i - r;
        if (y < 0 || y >= alto) continue;          // extensión con ceros
        for (int j = 0; j < k; ++j) {
            const int x = col + j - r;
            if (x < 0 || x >= ancho) continue;     // extensión con ceros
            suma += entrada[y * ancho + x] * mascara[i * k + j];
        }
    }
    salida[idx] = suma;
}

ConvolucionGPU::ConvolucionGPU(int ancho, int alto, int k) : ancho_(ancho), alto_(alto), k_(k) {
    const size_t bytesImg = (size_t)ancho * alto * sizeof(float);
    CUDA_OK(cudaMalloc(&dEntrada_, bytesImg));
    CUDA_OK(cudaMalloc(&dSalida_, bytesImg));
    CUDA_OK(cudaMalloc(&dMascara_, (size_t)k * k * sizeof(float)));
}

ConvolucionGPU::~ConvolucionGPU() {
    cudaFree(dEntrada_);
    cudaFree(dSalida_);
    cudaFree(dMascara_);
}

int ConvolucionGPU::ejecutar(const float* entrada, const float* mascara, float* salida,
                             int hilosPorBloque, TiemposGPU& t) {
    const size_t bytesImg = (size_t)ancho_ * alto_ * sizeof(float);
    const size_t bytesMasc = (size_t)k_ * k_ * sizeof(float);
    const int totalPixeles = ancho_ * alto_;
    // Bloques suficientes para cubrir todos los píxeles (redondeo hacia arriba).
    const int bloques = (totalPixeles + hilosPorBloque - 1) / hilosPorBloque;

    cudaEvent_t e0, e1, e2, e3;
    CUDA_OK(cudaEventCreate(&e0));
    CUDA_OK(cudaEventCreate(&e1));
    CUDA_OK(cudaEventCreate(&e2));
    CUDA_OK(cudaEventCreate(&e3));

    CUDA_OK(cudaEventRecord(e0));
    CUDA_OK(cudaMemcpy(dEntrada_, entrada, bytesImg, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dMascara_, mascara, bytesMasc, cudaMemcpyHostToDevice));
    CUDA_OK(cudaEventRecord(e1));

    convolucionKernel<<<bloques, hilosPorBloque>>>(dEntrada_, dSalida_, dMascara_, ancho_, alto_, k_);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaEventRecord(e2));

    CUDA_OK(cudaMemcpy(salida, dSalida_, bytesImg, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaEventRecord(e3));
    CUDA_OK(cudaEventSynchronize(e3));  // asegura que todo terminó antes de leer tiempos

    CUDA_OK(cudaEventElapsedTime(&t.h2d, e0, e1));
    CUDA_OK(cudaEventElapsedTime(&t.kernel, e1, e2));
    CUDA_OK(cudaEventElapsedTime(&t.d2h, e2, e3));
    CUDA_OK(cudaEventElapsedTime(&t.total, e0, e3));

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaEventDestroy(e2);
    cudaEventDestroy(e3);
    return bloques;
}

void imprimirInfoGPU() {
    int dispositivo = 0;
    cudaDeviceProp p;
    CUDA_OK(cudaGetDevice(&dispositivo));
    CUDA_OK(cudaGetDeviceProperties(&p, dispositivo));
    int runtime = 0, driver = 0;
    cudaRuntimeGetVersion(&runtime);
    cudaDriverGetVersion(&driver);
    std::printf("GPU: %s\n", p.name);
    std::printf("Capacidad de computo: %d.%d\n", p.major, p.minor);
    std::printf("Multiprocesadores (SM): %d\n", p.multiProcessorCount);
    std::printf("Memoria global: %.0f MiB\n", p.totalGlobalMem / (1024.0 * 1024.0));
    std::printf("Max hilos por bloque: %d\n", p.maxThreadsPerBlock);
    std::printf("Max bloques en X: %d\n", p.maxGridSize[0]);
    std::printf("CUDA runtime: %d.%d, driver: %d.%d\n", runtime / 1000, (runtime % 100) / 10,
                driver / 1000, (driver % 100) / 10);
}
