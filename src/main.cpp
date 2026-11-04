// Programa de medición: aplica un filtro a una imagen en CPU y en CUDA,
// valida que ambas salidas coincidan y guarda los tiempos en CSV.
//
// Uso:
//   convolucion --entrada data/img_4000.pgm --filtro media --k 9 --hilos 256
//               [--reps 5] [--csv resultados/tiempos.csv]
//               [--validacion resultados/validacion.csv]
//               [--salida resultados/imagenes/media9.pgm] [--sin-cpu]
//   convolucion --info

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "conv_cpu.h"
#include "conv_gpu.h"
#include "imagen.h"
#include "mascaras.h"
#include "validacion.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

// En CPUs híbridas (núcleos P y E, p. ej. Intel Core Ultra) Windows puede mover el hilo
// a un núcleo de eficiencia o reducir su frecuencia (EcoQoS) a mitad de la medición.
// Para que las repeticiones sean comparables: prioridad alta, sin limitación de energía
// y el hilo fijado a un núcleo de la clase de mayor rendimiento.
static void prepararCPUParaMedir() {
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    PROCESS_POWER_THROTTLING_STATE estado = {};
    estado.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
    estado.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
    estado.StateMask = 0;  // 0 = desactivar la limitación
    SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &estado, sizeof estado);

    ULONG largo = 0;
    GetSystemCpuSetInformation(nullptr, 0, &largo, GetCurrentProcess(), 0);
    std::vector<char> buf(largo);
    auto* info = reinterpret_cast<SYSTEM_CPU_SET_INFORMATION*>(buf.data());
    if (!GetSystemCpuSetInformation(info, largo, &largo, GetCurrentProcess(), 0)) return;

    ULONG mejorId = 0;
    int mejorClase = -1;
    for (ULONG pos = 0; pos < largo;) {
        auto* c = reinterpret_cast<SYSTEM_CPU_SET_INFORMATION*>(buf.data() + pos);
        if (c->Type == CpuSetInformation && c->CpuSet.EfficiencyClass > mejorClase) {
            mejorClase = c->CpuSet.EfficiencyClass;
            mejorId = c->CpuSet.Id;
        }
        pos += c->Size;
    }
    if (mejorClase >= 0) {
        SetThreadSelectedCpuSets(GetCurrentThread(), &mejorId, 1);
        std::printf("Hilo de CPU fijado al conjunto %lu (clase de eficiencia %d, la de mayor rendimiento)\n",
                    mejorId, mejorClase);
    }
}
#else
static void prepararCPUParaMedir() {}
#endif

struct Opciones {
    std::string entrada;
    std::string filtro = "media";
    int k = 3;
    int hilos = 256;
    int reps = 5;
    std::string csv = "resultados/tiempos.csv";
    std::string csvValidacion = "resultados/validacion.csv";
    std::string salida;
    bool conCPU = true;
    bool info = false;
};

static Opciones leerOpciones(int argc, char** argv) {
    Opciones o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto valor = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "Falta valor para %s\n", a.c_str()); std::exit(1); }
            return argv[++i];
        };
        if (a == "--entrada") o.entrada = valor();
        else if (a == "--filtro") o.filtro = valor();
        else if (a == "--k") o.k = std::atoi(valor().c_str());
        else if (a == "--hilos") o.hilos = std::atoi(valor().c_str());
        else if (a == "--reps") o.reps = std::atoi(valor().c_str());
        else if (a == "--csv") o.csv = valor();
        else if (a == "--validacion") o.csvValidacion = valor();
        else if (a == "--salida") o.salida = valor();
        else if (a == "--sin-cpu") o.conCPU = false;
        else if (a == "--info") o.info = true;
        else { std::fprintf(stderr, "Opcion desconocida: %s\n", a.c_str()); std::exit(1); }
    }
    return o;
}

// Abre el CSV en modo "agregar" y escribe la cabecera si el archivo está vacío.
static std::ofstream abrirCSV(const std::string& ruta, const char* cabecera) {
    bool nuevo;
    { std::ifstream prueba(ruta); nuevo = !prueba.good() || prueba.peek() == EOF; }
    std::ofstream f(ruta, std::ios::app);
    if (!f) { std::fprintf(stderr, "No se pudo abrir %s\n", ruta.c_str()); std::exit(1); }
    if (nuevo) f << cabecera << "\n";
    return f;
}

int main(int argc, char** argv) {
    Opciones o = leerOpciones(argc, argv);
    if (o.info) { imprimirInfoGPU(); return 0; }
    if (o.entrada.empty()) { std::fprintf(stderr, "Falta --entrada\n"); return 1; }

    // --- Preparación (no se mide) ---
    prepararCPUParaMedir();
    Imagen img = leerPGM(o.entrada);
    Mascara m = crearMascara(o.filtro, o.k);
    const size_t n = (size_t)img.ancho * img.alto;
    std::vector<float> salidaCPU(n), salidaGPU(n);
    ConvolucionGPU gpu(img.ancho, img.alto, o.k);  // reserva de memoria fuera de la medición

    std::printf("Imagen %dx%d | filtro %s %dx%d | %d hilos/bloque | %d repeticiones\n",
                img.ancho, img.alto, o.filtro.c_str(), o.k, o.k, o.hilos, o.reps);

    std::ofstream csv = abrirCSV(o.csv,
        "implementacion,filtro,ancho,alto,k,hilos_por_bloque,bloques,repeticion,"
        "t_h2d,t_computo,t_d2h,t_total,unidad");

    // --- CPU: 1 calentamiento + reps medidas ---
    if (o.conCPU) {
        convolucionCPU(img.datos.data(), salidaCPU.data(), img.ancho, img.alto, m.coef.data(), o.k);
        for (int r = 1; r <= o.reps; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            convolucionCPU(img.datos.data(), salidaCPU.data(), img.ancho, img.alto, m.coef.data(), o.k);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            csv << "CPU," << o.filtro << "," << img.ancho << "," << img.alto << "," << o.k
                << ",0,0," << r << ",0," << ms << ",0," << ms << ",ms\n";
            std::printf("  CPU  rep %d: %.3f ms\n", r, ms);
        }
    }

    // --- GPU: 3 calentamientos + reps medidas ---
    // Con uno no basta: tras la fase de CPU la GPU del portátil queda en reposo y
    // necesita varias ejecuciones para subir su frecuencia de reloj.
    TiemposGPU t;
    int bloques = 0;
    for (int c = 0; c < 3; ++c)
        bloques = gpu.ejecutar(img.datos.data(), m.coef.data(), salidaGPU.data(), o.hilos, t);
    for (int r = 1; r <= o.reps; ++r) {
        gpu.ejecutar(img.datos.data(), m.coef.data(), salidaGPU.data(), o.hilos, t);
        csv << "CUDA," << o.filtro << "," << img.ancho << "," << img.alto << "," << o.k << ","
            << o.hilos << "," << bloques << "," << r << "," << t.h2d << "," << t.kernel << ","
            << t.d2h << "," << t.total << ",ms\n";
        std::printf("  CUDA rep %d: h2d %.3f | kernel %.3f | d2h %.3f | total %.3f ms\n",
                    r, t.h2d, t.kernel, t.d2h, t.total);
    }

    // --- Validación numérica CPU vs GPU (antes de convertir a [0,255]) ---
    if (o.conCPU) {
        ErrorNumerico e = compararSalidas(salidaCPU.data(), salidaGPU.data(), n);
        const bool ok = e.maximo <= TOLERANCIA;
        std::printf("  Validacion: error max %.3e | error medio %.3e | tolerancia %.0e -> %s\n",
                    e.maximo, e.medio, TOLERANCIA, ok ? "OK" : "FALLA");
        std::ofstream val = abrirCSV(o.csvValidacion,
            "filtro,ancho,alto,k,hilos_por_bloque,error_max,error_medio,tolerancia,resultado");
        val << o.filtro << "," << img.ancho << "," << img.alto << "," << o.k << "," << o.hilos << ","
            << e.maximo << "," << e.medio << "," << TOLERANCIA << "," << (ok ? "OK" : "FALLA") << "\n";
        if (!ok) return 2;
    }

    // --- Imagen de salida (solo para el informe, no se mide) ---
    if (!o.salida.empty()) {
        escribirPGM(o.salida, img.ancho, img.alto, aVisualizacion(salidaGPU, o.filtro));
        std::printf("  Salida guardada en %s\n", o.salida.c_str());
    }
    return 0;
}
