// Convolución secuencial en CPU (versión de referencia).
#pragma once

// entrada y salida son arreglos distintos de ancho*alto floats.
// mascara tiene k*k coeficientes. Fuera de la imagen se asume valor 0.
void convolucionCPU(const float* entrada, float* salida, int ancho, int alto,
                    const float* mascara, int k);
