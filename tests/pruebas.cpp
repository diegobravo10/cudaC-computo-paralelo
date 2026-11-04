// Pruebas de corrección:
//  1. Matriz 5x5 calculada a mano (incluye bordes con extensión de ceros).
//  2. Imágenes cuyas dimensiones NO son múltiplos de los hilos por bloque.
//  3. Casos extremos: imagen 1x1, una sola fila, una sola columna, máscara mayor que la imagen.

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "../src/conv_cpu.h"
#include "../src/conv_gpu.h"
#include "../src/mascaras.h"
#include "../src/validacion.h"

static int fallos = 0;

static void reportar(const std::string& nombre, bool ok, const std::string& detalle = "") {
    std::printf("[%s] %s %s\n", ok ? " OK  " : "FALLA", nombre.c_str(), detalle.c_str());
    if (!ok) ++fallos;
}

static std::vector<float> ejecutarGPU(const std::vector<float>& entrada, int ancho, int alto,
                                      const Mascara& m, int hilos) {
    std::vector<float> salida(entrada.size());
    ConvolucionGPU gpu(ancho, alto, m.k);
    TiemposGPU t;
    gpu.ejecutar(entrada.data(), m.coef.data(), salida.data(), hilos, t);
    return salida;
}

static std::vector<float> ejecutarCPU(const std::vector<float>& entrada, int ancho, int alto,
                                      const Mascara& m) {
    std::vector<float> salida(entrada.size());
    convolucionCPU(entrada.data(), salida.data(), ancho, alto, m.coef.data(), m.k);
    return salida;
}

// ---------------------------------------------------------------------------
// 1. Prueba manual. Entrada: f(fila, col) = 5*fila + col + 1
//     1  2  3  4  5
//     6  7  8  9 10
//    11 12 13 14 15
//    16 17 18 19 20
//    21 22 23 24 25
// Los valores esperados se calcularon a mano (ver DESARROLLO.md).
// ---------------------------------------------------------------------------
static void pruebaManual() {
    std::vector<float> entrada(25);
    for (int i = 0; i < 25; ++i) entrada[i] = (float)(i + 1);

    const std::vector<float> esperadoLaplaciano = {
          4,   3,   2,   1,  -6,
         -5,   0,   0,   0, -11,
        -10,   0,   0,   0, -16,
        -15,   0,   0,   0, -21,
        -46, -27, -28, -29, -56 };
    const std::vector<float> esperadoNitidez = {
         -3,  -1,   1,   3,  11,
         11,   7,   8,   9,  21,
         21,  12,  13,  14,  31,
         31,  17,  18,  19,  41,
         67,  49,  51,  53,  81 };
    // Media 3x3: se comprueban algunas posiciones representativas.
    struct Punto { int fila, col; float valor; };
    const std::vector<Punto> esperadoMedia = {
        {2, 2, 117.0f / 9},  // centro: 7+8+9+12+13+14+17+18+19
        {0, 0, 16.0f / 9},   // esquina: 1+2+6+7
        {0, 2, 33.0f / 9},   // borde superior: 2+3+4+7+8+9
        {4, 4, 88.0f / 9},   // esquina inferior derecha: 19+20+24+25
    };

    auto comprobar = [&](const std::string& filtro, const std::vector<float>& esperado) {
        Mascara m = crearMascara(filtro, 3);
        auto cpu = ejecutarCPU(entrada, 5, 5, m);
        auto gpu = ejecutarGPU(entrada, 5, 5, m, 128);
        ErrorNumerico eCpu = compararSalidas(cpu.data(), esperado.data(), 25);
        ErrorNumerico eGpu = compararSalidas(gpu.data(), esperado.data(), 25);
        char buf[128];
        std::snprintf(buf, sizeof buf, "(err max CPU %.1e, GPU %.1e)", eCpu.maximo, eGpu.maximo);
        reportar("manual 5x5 " + filtro, eCpu.maximo <= 1e-5 && eGpu.maximo <= 1e-5, buf);
    };
    comprobar("laplaciano", esperadoLaplaciano);
    comprobar("nitidez", esperadoNitidez);

    Mascara media = crearMascara("media", 3);
    auto cpu = ejecutarCPU(entrada, 5, 5, media);
    auto gpu = ejecutarGPU(entrada, 5, 5, media, 128);
    double errMax = 0;
    for (const Punto& p : esperadoMedia) {
        errMax = std::fmax(errMax, std::fabs(cpu[p.fila * 5 + p.col] - p.valor));
        errMax = std::fmax(errMax, std::fabs(gpu[p.fila * 5 + p.col] - p.valor));
    }
    char buf[64];
    std::snprintf(buf, sizeof buf, "(err max %.1e)", errMax);
    reportar("manual 5x5 media", errMax <= 1e-5, buf);
}

// ---------------------------------------------------------------------------
// 2 y 3. CPU vs GPU con dimensiones no múltiplo de los hilos por bloque.
// ---------------------------------------------------------------------------
static void pruebaDimensiones(int ancho, int alto, const std::string& filtro, int k, int hilos) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<float> entrada((size_t)ancho * alto);
    for (float& v : entrada) v = (float)dist(gen);

    Mascara m = crearMascara(filtro, k);
    auto cpu = ejecutarCPU(entrada, ancho, alto, m);
    auto gpu = ejecutarGPU(entrada, ancho, alto, m, hilos);
    ErrorNumerico e = compararSalidas(cpu.data(), gpu.data(), cpu.size());

    char nombre[128], detalle[128];
    std::snprintf(nombre, sizeof nombre, "%dx%d %s k=%d hilos=%d (pixeles %% hilos = %d)",
                  ancho, alto, filtro.c_str(), k, hilos, (ancho * alto) % hilos);
    std::snprintf(detalle, sizeof detalle, "(err max %.2e, medio %.2e)", e.maximo, e.medio);
    reportar(nombre, e.maximo <= TOLERANCIA, detalle);
}

int main() {
    std::printf("== Prueba manual ==\n");
    pruebaManual();

    std::printf("\n== Dimensiones no multiplo de hilos por bloque ==\n");
    for (int hilos : {128, 256, 512}) {
        pruebaDimensiones(1001, 777, "media", 3, hilos);
        pruebaDimensiones(1001, 777, "nitidez", 3, hilos);
        pruebaDimensiones(1001, 777, "laplaciano", 3, hilos);
        pruebaDimensiones(1001, 777, "media", 9, hilos);
        pruebaDimensiones(1001, 777, "media", 21, hilos);
    }

    std::printf("\n== Casos extremos ==\n");
    pruebaDimensiones(1, 1, "media", 3, 256);
    pruebaDimensiones(257, 1, "laplaciano", 3, 256);
    pruebaDimensiones(1, 257, "nitidez", 3, 256);
    pruebaDimensiones(7, 5, "media", 21, 128);  // máscara más grande que la imagen

    std::printf("\n%s (%d fallos)\n", fallos ? "HAY FALLOS" : "TODAS LAS PRUEBAS PASARON", fallos);
    return fallos ? 1 : 0;
}
