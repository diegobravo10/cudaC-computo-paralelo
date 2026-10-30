// Lectura/escritura de imágenes en escala de grises (formato PGM binario, P5)
// y conversión de resultados al rango de visualización [0, 255].
#pragma once

#include <string>
#include <vector>

struct Imagen {
    int ancho = 0;
    int alto = 0;
    std::vector<float> datos;  // fila por fila: datos[fila * ancho + col]
};

// Lee un PGM de 8 bits y convierte cada píxel a float en [0, 255].
Imagen leerPGM(const std::string& ruta);

// Escribe un PGM de 8 bits a partir de valores ya convertidos a [0, 255].
void escribirPGM(const std::string& ruta, int ancho, int alto, const std::vector<unsigned char>& datos);

// Convierte la salida float de la convolución a bytes para visualizar.
//  - media y nitidez: redondeo y recorte a [0, 255]
//  - laplaciano: valor absoluto (magnitud del cambio), redondeo y recorte a [0, 255]
std::vector<unsigned char> aVisualizacion(const std::vector<float>& valores, const std::string& filtro);
