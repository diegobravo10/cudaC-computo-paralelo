#include "conv_cpu.h"

// Se aplica la máscara sin invertirla (correlación). Como todas las máscaras
// de la práctica son simétricas, el resultado es idéntico a la convolución.
// El orden de las sumas (i luego j) es el mismo que en el kernel CUDA, para que
// las diferencias numéricas CPU-GPU sean mínimas.
void convolucionCPU(const float* entrada, float* salida, int ancho, int alto,
                    const float* mascara, int k) {
    const int r = k / 2;
    for (int fila = 0; fila < alto; ++fila) {
        for (int col = 0; col < ancho; ++col) {
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
            salida[fila * ancho + col] = suma;
        }
    }
}
