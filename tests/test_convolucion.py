import unittest

import numpy as np

from src.convolucion import (ConvolucionGPU, TOLERANCIA,
                             TiempoSecuencialExcedido,
                             convolucion_secuencial, crear_mascara)


class PruebasConvolucion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import pycuda.driver as cuda
            cuda.init()
            if cuda.Device.count() == 0:
                raise unittest.SkipTest("No hay una GPU CUDA disponible")
        except Exception as exc:
            raise unittest.SkipTest(f"PyCUDA/CUDA no disponible: {exc}") from exc

    def comparar(self, entrada, filtro, k, hilos=128):
        entrada = np.asarray(entrada, dtype=np.float32)
        mascara = crear_mascara(filtro, k)
        cpu = convolucion_secuencial(entrada, mascara)
        gpu, bloques, _ = ConvolucionGPU(entrada.shape[1], entrada.shape[0], k).ejecutar(
            entrada, mascara, hilos)
        self.assertEqual(bloques, (entrada.size + hilos - 1) // hilos)
        np.testing.assert_allclose(gpu, cpu, atol=TOLERANCIA, rtol=0)
        return cpu, gpu

    def test_manual_5x5_laplaciano(self):
        esperado = np.array([[4, 3, 2, 1, -6], [-5, 0, 0, 0, -11],
                             [-10, 0, 0, 0, -16], [-15, 0, 0, 0, -21],
                             [-46, -27, -28, -29, -56]], dtype=np.float32)
        cpu, gpu = self.comparar(np.arange(1, 26).reshape(5, 5), "laplaciano", 3)
        np.testing.assert_allclose(cpu, esperado, atol=1e-5)
        np.testing.assert_allclose(gpu, esperado, atol=1e-5)

    def test_manual_5x5_nitidez(self):
        esperado = np.array([[-3, -1, 1, 3, 11], [11, 7, 8, 9, 21],
                             [21, 12, 13, 14, 31], [31, 17, 18, 19, 41],
                             [67, 49, 51, 53, 81]], dtype=np.float32)
        cpu, gpu = self.comparar(np.arange(1, 26).reshape(5, 5), "nitidez", 3)
        np.testing.assert_allclose(cpu, esperado, atol=1e-5)
        np.testing.assert_allclose(gpu, esperado, atol=1e-5)

    def test_dimensiones_no_multiplo(self):
        # 17*19=323, no divisible para 128 ni 256; prueba el último bloque parcial.
        entrada = np.random.default_rng(42).integers(0, 256, (17, 19)).astype(np.float32)
        for hilos in (128, 256):
            for filtro, k in (("media", 3), ("nitidez", 3), ("laplaciano", 3),
                              ("media", 9), ("media", 21)):
                with self.subTest(hilos=hilos, filtro=filtro, k=k):
                    self.comparar(entrada, filtro, k, hilos)

    def test_casos_extremos(self):
        for forma, filtro, k in (((1, 1), "media", 3), ((1, 257), "laplaciano", 3),
                                 ((257, 1), "nitidez", 3), ((5, 7), "media", 21)):
            with self.subTest(forma=forma, filtro=filtro):
                self.comparar(np.arange(np.prod(forma)).reshape(forma), filtro, k)

    def test_limite_secuencial_documentado(self):
        with self.assertRaises(TiempoSecuencialExcedido):
            convolucion_secuencial(np.ones((100, 100), dtype=np.float32),
                                   crear_mascara("media", 21), limite_segundos=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
