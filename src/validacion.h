// Comparación numérica entre dos salidas (antes de convertir a [0, 255]).
#pragma once

#include <cmath>
#include <cstddef>

// Tolerancia absoluta para aceptar que CPU y GPU coinciden.
// Justificación en DESARROLLO.md (sección "Validación").
constexpr double TOLERANCIA = 1e-3;

struct ErrorNumerico {
    double maximo = 0;
    double medio = 0;
};

inline ErrorNumerico compararSalidas(const float* a, const float* b, size_t n) {
    ErrorNumerico e;
    double suma = 0;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs((double)a[i] - (double)b[i]);
        suma += d;
        if (d > e.maximo) e.maximo = d;
    }
    e.medio = n ? suma / n : 0;
    return e;
}
