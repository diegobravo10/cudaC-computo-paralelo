// Convolución en GPU con CUDA: bloques e hilos en una sola dimensión (X).
#pragma once

#include <string>

// Tiempos de una ejecución, en milisegundos (medidos con eventos CUDA).
struct TiemposGPU {
    float h2d = 0;     // copia CPU -> GPU (imagen + máscara)
    float kernel = 0;  // ejecución del kernel
    float d2h = 0;     // copia GPU -> CPU (resultado)
    float total = 0;   // desde el inicio de h2d hasta el fin de d2h
};

class ConvolucionGPU {
public:
    // Reserva la memoria en la GPU una sola vez (fuera de la medición).
    ConvolucionGPU(int ancho, int alto, int k);
    ~ConvolucionGPU();

    // Copia la entrada, ejecuta el kernel y trae el resultado.
    // Devuelve el número de bloques usados.
    int ejecutar(const float* entrada, const float* mascara, float* salida,
                 int hilosPorBloque, TiemposGPU& tiempos);

private:
    int ancho_, alto_, k_;
    float* dEntrada_ = nullptr;
    float* dSalida_ = nullptr;
    float* dMascara_ = nullptr;
};

// Imprime los datos de la GPU (nombre, memoria, capacidad de cómputo...).
void imprimirInfoGPU();

// Aborta con mensaje si una llamada CUDA falla.
void verificarCuda(int codigo, const char* archivo, int linea);
#define CUDA_OK(x) verificarCuda((int)(x), __FILE__, __LINE__)
