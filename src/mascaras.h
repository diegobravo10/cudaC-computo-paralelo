// Máscaras de convolución usadas en la práctica.
#pragma once

#include <string>
#include <vector>

struct Mascara {
    int k = 0;                  // tamaño k x k (k impar)
    std::vector<float> coef;    // coef[i * k + j], i = fila, j = columna
};

// filtro: "media" (k impar cualquiera), "nitidez" (solo k = 3), "laplaciano" (solo k = 3)
Mascara crearMascara(const std::string& filtro, int k);
