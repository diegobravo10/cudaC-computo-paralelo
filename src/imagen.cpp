#include "imagen.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

// Salta espacios y comentarios (#...) de la cabecera PGM.
static void saltarEspacios(std::istream& in) {
    while (true) {
        int c = in.peek();
        if (c == '#') {
            std::string linea;
            std::getline(in, linea);
        } else if (std::isspace(c)) {
            in.get();
        } else {
            break;
        }
    }
}

Imagen leerPGM(const std::string& ruta) {
    std::ifstream in(ruta, std::ios::binary);
    if (!in) throw std::runtime_error("No se pudo abrir " + ruta);

    std::string magia;
    in >> magia;
    if (magia != "P5") throw std::runtime_error(ruta + " no es un PGM binario (P5)");

    int ancho, alto, maxval;
    saltarEspacios(in); in >> ancho;
    saltarEspacios(in); in >> alto;
    saltarEspacios(in); in >> maxval;
    in.get();  // un único espacio en blanco antes de los datos
    if (maxval != 255) throw std::runtime_error("Solo se admiten PGM de 8 bits (maxval 255)");

    std::vector<unsigned char> bytes((size_t)ancho * alto);
    in.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if ((size_t)in.gcount() != bytes.size()) throw std::runtime_error("PGM incompleto: " + ruta);

    Imagen img;
    img.ancho = ancho;
    img.alto = alto;
    img.datos.assign(bytes.begin(), bytes.end());
    return img;
}

void escribirPGM(const std::string& ruta, int ancho, int alto, const std::vector<unsigned char>& datos) {
    std::ofstream out(ruta, std::ios::binary);
    if (!out) throw std::runtime_error("No se pudo crear " + ruta);
    out << "P5\n" << ancho << " " << alto << "\n255\n";
    out.write(reinterpret_cast<const char*>(datos.data()), datos.size());
}

std::vector<unsigned char> aVisualizacion(const std::vector<float>& valores, const std::string& filtro) {
    const bool absoluto = (filtro == "laplaciano");
    std::vector<unsigned char> salida(valores.size());
    for (size_t i = 0; i < valores.size(); ++i) {
        float v = absoluto ? std::fabs(valores[i]) : valores[i];
        v = std::round(v);
        salida[i] = (unsigned char)std::clamp(v, 0.0f, 255.0f);
    }
    return salida;
}
