(() => {
  const formulario = document.getElementById('formulario');
  const imagen = document.getElementById('imagen');
  const filtro = document.getElementById('filtro');
  const mascara = document.getElementById('mascara');
  const hilos = document.getElementById('hilos');
  const enviar = document.getElementById('enviar');
  const textoEnvio = document.getElementById('texto-envio');
  const iconoEnvio = document.getElementById('icono-envio');
  const progreso = document.getElementById('progreso');
  const alerta = document.getElementById('alerta');
  const resultados = document.getElementById('resultados');
  const original = document.getElementById('original');
  const procesada = document.getElementById('procesada');
  const descargar = document.getElementById('descargar');
  const crudo = document.getElementById('crudo');
  const estadoSalud = document.getElementById('estado-salud');
  const requestId = document.getElementById('request-id');
  const arranque = document.getElementById('arranque');

  const camposTiempo = {
    queue_wait_ms: 't-cola',
    cpu_prep_ms: 't-prep',
    h2d_ms: 't-h2d',
    kernel_ms: 't-kernel',
    d2h_ms: 't-d2h',
    postprocess_ms: 't-post',
    server_total_ms: 't-total',
  };

  let enviando = false;
  let urlOriginal = null;
  let urlProcesada = null;

  const textos = {
    't-cola': 'espera en cola',
    't-prep': 'preparación CPU',
    't-h2d': 'reserva + H2D',
    't-kernel': 'kernel CUDA',
    't-d2h': 'D2H',
    't-post': 'postproceso',
    't-total': 'total en servidor',
  };

  function mostrarError(mensaje) {
    alerta.textContent = mensaje;
    alerta.hidden = false;
  }

  function limpiarError() {
    alerta.hidden = true;
    alerta.textContent = '';
  }

  function actualizarMascara() {
    const esMedia = filtro.value === 'media';
    mascara.disabled = !esMedia;
    if (!esMedia) mascara.value = '3';
  }

  function establecerEnvio(activo) {
    enviando = activo;
    enviar.disabled = activo;
    iconoEnvio.hidden = !activo;
    progreso.hidden = !activo;
    textoEnvio.textContent = activo ? 'Procesando…' : 'Procesar en GPU';
  }

  function liberarUrls() {
    if (urlOriginal) URL.revokeObjectURL(urlOriginal);
    if (urlProcesada) URL.revokeObjectURL(urlProcesada);
    urlOriginal = null;
    urlProcesada = null;
  }

  function pintarTiempos(metricas) {
    for (const [clave, id] of Object.entries(camposTiempo)) {
      const valor = metricas[clave];
      document.getElementById(id).textContent =
        `${Number(valor).toFixed(3)} ms · ${textos[id]}`;
    }
    arranque.textContent = metricas.cold_start
      ? 'Arranque en frío: contextualización y compilación incluidas en esta solicitud.'
      : 'Ejecución estable: contexto y kernel reutilizados (arranque en frío excluido).';
  }

  async function verificarSalud() {
    try {
      const respuesta = await fetch('/api/health');
      const datos = await respuesta.json();
      if (respuesta.ok && datos.status === 'ok') {
        estadoSalud.textContent = `${datos.gpu.dispositivo} · ${datos.gpu.pycuda}`;
        estadoSalud.className = 'badge ok';
      } else {
        throw new Error(datos.error?.message || 'GPU no disponible');
      }
    } catch (error) {
      estadoSalud.textContent = `GPU no disponible: ${error.message}`;
      estadoSalud.className = 'badge error';
    }
  }

  filtro.addEventListener('change', actualizarMascara);

  formulario.addEventListener('submit', async (evento) => {
    evento.preventDefault();
    if (enviando) return;
    limpiarError();
    if (!imagen.files || imagen.files.length === 0) {
      mostrarError('Seleccione una imagen antes de enviar la solicitud.');
      return;
    }

    establecerEnvio(true);
    const datos = new FormData();
    datos.append('file', imagen.files[0]);
    datos.append('filter', filtro.value);
    datos.append('mask_size', mascara.value);
    datos.append('threads_per_block', hilos.value);

    try {
      const respuesta = await fetch('/api/convolve', { method: 'POST', body: datos });
      const contenido = await respuesta.json();
      if (!respuesta.ok) {
        const detalle = contenido.error || {};
        mostrarError(`${detalle.code || 'ERROR'}: ${detalle.message || 'No se pudo procesar'}`);
        return;
      }

      const resultado = await fetch(contenido.result_url);
      if (!resultado.ok) throw new Error('No se pudo descargar la imagen procesada');
      const blob = await resultado.blob();

      liberarUrls();
      urlOriginal = URL.createObjectURL(imagen.files[0]);
      urlProcesada = URL.createObjectURL(blob);
      original.src = urlOriginal;
      procesada.src = urlProcesada;
      descargar.href = urlProcesada;
      descargar.download = contenido.download_filename;
      crudo.href = contenido.raw_url;
      crudo.download = contenido.raw_url.split('/').pop();
      requestId.textContent = contenido.request_id;
      pintarTiempos(contenido.metrics);
      resultados.hidden = false;
    } catch (error) {
      mostrarError(`Error de comunicación: ${error.message}`);
    } finally {
      establecerEnvio(false);
    }
  });

  actualizarMascara();
  verificarSalud();
})();
