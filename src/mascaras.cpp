#include "mascaras.h"

#include <stdexcept>

Mascara crearMascara(const std::string& filtro, int k) {
    Mascara m;
    m.k = k;

    if (filtro == "media") {
        if (k < 1 || k % 2 == 0) throw std::runtime_error("La media necesita k impar >= 1");
        // Todos los coeficientes valen 1/(k*k): promedio de la vecindad.
        m.coef.assign((size_t)k * k, 1.0f / (float)(k * k));
    } else if (filtro == "nitidez") {
        if (k != 3) throw std::runtime_error("Nitidez solo está definida para k = 3");
        m.coef = { 0, -1,  0,
                  -1,  5, -1,
                   0, -1,  0 };
    } else if (filtro == "laplaciano") {
        if (k != 3) throw std::runtime_error("Laplaciano solo está definido para k = 3");
        m.coef = { 0,  1, 0,
                   1, -4, 1,
                   0,  1, 0 };
    } else {
        throw std::runtime_error("Filtro desconocido: " + filtro);
    }
    return m;
}
